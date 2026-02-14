"""Scheduling engine for generating posting recommendations."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..models.engagement_predictor import EngagementPredictor
from ..optimization.contextual_bandit import PostingTimeOptimizer
from ..features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class PostingScheduler:
    """Main scheduling engine that combines prediction and optimization."""
    
    def __init__(self, predictor: EngagementPredictor, 
                 optimizer: PostingTimeOptimizer,
                 feature_engineer: FeatureEngineer):
        """Initialize posting scheduler.
        
        Args:
            predictor: Trained engagement prediction model
            optimizer: Contextual bandit optimizer
            feature_engineer: Fitted feature engineering pipeline
        """
        self.predictor = predictor
        self.optimizer = optimizer
        self.feature_engineer = feature_engineer
        
    def generate_daily_schedule(self, date: datetime, context: Dict,
                              top_k: int = 3) -> List[Dict]:
        """Generate posting schedule for a specific date.
        
        Args:
            date: Target date for scheduling
            context: Context features for the date
            top_k: Number of recommendations to generate
            
        Returns:
            List of posting recommendations
        """
        logger.info(f"Generating schedule for {date.strftime('%Y-%m-%d')}")
        
        # Create context for the target date
        full_context = context.copy()
        full_context.update({
            'day_of_week': date.weekday(),
            'is_weekend': 1 if date.weekday() >= 5 else 0,
            'day_of_month': date.day,
            'month': date.month
        })
        
        # Get bandit recommendations
        bandit_recommendations = self.optimizer.recommend_posting_times(
            full_context, top_k=top_k
        )
        
        # Enhance recommendations with engagement predictions
        enhanced_recommendations = []
        
        for rec in bandit_recommendations:
            hour = rec['hour_of_day']
            
            # Create prediction context for this hour
            pred_context = full_context.copy()
            pred_context['hour_of_day'] = hour
            
            # Create dummy DataFrame for prediction
            pred_df = pd.DataFrame([pred_context])
            
            # Generate dummy embeddings (in practice, use actual title embeddings)
            dummy_embeddings = np.zeros((1, 384))  # Assuming 384-dim embeddings
            
            try:
                # Get engagement prediction with confidence
                pred_result = self.predictor.predict_with_confidence(
                    pred_df, dummy_embeddings
                )
                
                expected_views = pred_result['predictions'][0]
                uncertainty = pred_result['uncertainty'][0]
                
            except Exception as e:
                logger.warning(f"Prediction failed for hour {hour}: {e}")
                expected_views = 0
                uncertainty = 0
            
            # Combine bandit confidence with prediction uncertainty
            combined_confidence = rec['confidence_score'] * (1 - uncertainty / expected_views if expected_views > 0 else 0.5)
            
            enhanced_rec = {
                'date': date.strftime('%Y-%m-%d'),
                'hour_of_day': hour,
                'time_display': rec['time_display'],
                'expected_views': int(expected_views),
                'bandit_confidence': rec['confidence_score'],
                'prediction_uncertainty': uncertainty,
                'combined_confidence': combined_confidence,
                'rank': rec['rank'],
                'recommendation_type': 'bandit_enhanced'
            }
            
            enhanced_recommendations.append(enhanced_rec)
        
        # Sort by combined confidence
        enhanced_recommendations.sort(key=lambda x: x['combined_confidence'], reverse=True)
        
        # Update ranks
        for i, rec in enumerate(enhanced_recommendations):
            rec['rank'] = i + 1
        
        logger.info(f"Generated {len(enhanced_recommendations)} recommendations for {date.strftime('%Y-%m-%d')}")
        
        return enhanced_recommendations
    
    def generate_weekly_schedule(self, start_date: datetime, context: Dict,
                               top_k_per_day: int = 3) -> Dict[str, List[Dict]]:
        """Generate posting schedule for a week.
        
        Args:
            start_date: Start date of the week
            context: Base context features
            top_k_per_day: Number of recommendations per day
            
        Returns:
            Dictionary mapping dates to recommendation lists
        """
        weekly_schedule = {}
        
        for i in range(7):
            date = start_date + timedelta(days=i)
            daily_schedule = self.generate_daily_schedule(
                date, context, top_k=top_k_per_day
            )
            weekly_schedule[date.strftime('%Y-%m-%d')] = daily_schedule
        
        return weekly_schedule
    
    def create_heatmap_visualization(self, context: Dict, 
                                   save_path: Optional[str] = None) -> go.Figure:
        """Create weekly posting heatmap visualization.
        
        Args:
            context: Base context features
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Get heatmap data from optimizer
        heatmap_df = self.optimizer.get_weekly_heatmap_data(context)
        
        # Pivot for heatmap format
        heatmap_pivot = heatmap_df.pivot(
            index='hour_of_day', 
            columns='day_of_week', 
            values='confidence_score'
        )
        
        # Day names for better display
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        heatmap_pivot.columns = day_names
        
        # Create Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=day_names,
            y=list(range(24)),
            colorscale='Viridis',
            colorbar=dict(title="Confidence Score"),
            hoverongap=0
        ))
        
        fig.update_layout(
            title='Weekly Posting Time Confidence Heatmap',
            xaxis_title='Day of Week',
            yaxis_title='Hour of Day',
            yaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=2,
                autorange='reversed'
            ),
            height=600,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Heatmap saved to {save_path}")
        
        return fig
    
    def create_performance_dashboard(self, performance_data: pd.DataFrame,
                                   save_path: Optional[str] = None) -> go.Figure:
        """Create performance tracking dashboard.
        
        Args:
            performance_data: DataFrame with historical performance
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Views Over Time', 'Hourly Performance', 
                          'Day of Week Performance', 'Engagement Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Views over time
        fig.add_trace(
            go.Scatter(
                x=performance_data['published_at'],
                y=performance_data['views'],
                mode='lines+markers',
                name='Views',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Hourly performance
        hourly_avg = performance_data.groupby('hour_of_day')['views'].mean()
        fig.add_trace(
            go.Bar(
                x=hourly_avg.index,
                y=hourly_avg.values,
                name='Avg Views by Hour',
                marker_color='green'
            ),
            row=1, col=2
        )
        
        # Day of week performance
        dow_avg = performance_data.groupby('day_of_week')['views'].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig.add_trace(
            go.Bar(
                x=day_names,
                y=dow_avg.values,
                name='Avg Views by Day',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # Engagement rate over time
        fig.add_trace(
            go.Scatter(
                x=performance_data['published_at'],
                y=performance_data['engagement_rate'],
                mode='lines+markers',
                name='Engagement Rate',
                line=dict(color='red')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Performance Dashboard"
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig
    
    def evaluate_recommendations(self, recommendations: List[Dict],
                               actual_performance: Dict) -> Dict:
        """Evaluate recommendation quality against actual performance.
        
        Args:
            recommendations: List of recommendations made
            actual_performance: Dictionary with actual performance metrics
            
        Returns:
            Evaluation metrics dictionary
        """
        metrics = {
            'total_recommendations': len(recommendations),
            'avg_confidence': np.mean([r['combined_confidence'] for r in recommendations]),
            'top_recommendation_accuracy': 0,
            'recommendation_correlation': 0
        }
        
        if actual_performance:
            # Check if top recommendation was accurate
            top_rec = recommendations[0] if recommendations else None
            if top_rec and 'actual_views' in actual_performance:
                predicted_views = top_rec['expected_views']
                actual_views = actual_performance['actual_views']
                
                # Calculate accuracy (inverse of relative error)
                if predicted_views > 0:
                    relative_error = abs(actual_views - predicted_views) / predicted_views
                    metrics['top_recommendation_accuracy'] = max(0, 1 - relative_error)
        
        return metrics
    
    def update_with_feedback(self, recommendation: Dict, actual_performance: Dict,
                           context: Dict) -> None:
        """Update models with actual performance feedback.
        
        Args:
            recommendation: The recommendation that was used
            actual_performance: Actual performance metrics
            context: Context features used for recommendation
        """
        hour = recommendation['hour_of_day']
        actual_views = actual_performance.get('actual_views', 0)
        baseline_views = actual_performance.get('baseline_views')
        
        # Update bandit optimizer
        self.optimizer.update_performance(context, hour, actual_views, baseline_views)
        
        logger.info(f"Updated models with feedback for hour {hour}: {actual_views} views")
    
    def export_schedule(self, schedule: Dict, filepath: str, format: str = 'json') -> None:
        """Export schedule to file.
        
        Args:
            schedule: Schedule dictionary
            filepath: Output file path
            format: Export format ('json', 'csv')
        """
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(schedule, f, indent=2, default=str)
                
        elif format == 'csv':
            # Flatten schedule for CSV export
            rows = []
            for date, recommendations in schedule.items():
                for rec in recommendations:
                    row = {'date': date}
                    row.update(rec)
                    rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
        
        logger.info(f"Schedule exported to {filepath}")
    
    def get_schedule_summary(self, schedule: Dict) -> Dict:
        """Get summary statistics for a schedule.
        
        Args:
            schedule: Schedule dictionary
            
        Returns:
            Summary statistics dictionary
        """
        all_recommendations = []
        for recommendations in schedule.values():
            all_recommendations.extend(recommendations)
        
        if not all_recommendations:
            return {'total_recommendations': 0}
        
        summary = {
            'total_recommendations': len(all_recommendations),
            'avg_expected_views': np.mean([r['expected_views'] for r in all_recommendations]),
            'avg_confidence': np.mean([r['combined_confidence'] for r in all_recommendations]),
            'most_recommended_hours': {},
            'confidence_distribution': {}
        }
        
        # Most recommended hours
        hour_counts = {}
        for rec in all_recommendations:
            hour = rec['hour_of_day']
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        summary['most_recommended_hours'] = dict(
            sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        
        # Confidence distribution
        confidences = [r['combined_confidence'] for r in all_recommendations]
        summary['confidence_distribution'] = {
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences))
        }
        
        return summary