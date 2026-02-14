#!/usr/bin/env python3
"""Generate posting schedule recommendations."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
from datetime import datetime, timedelta
import json

from src.utils.config import Config, setup_logging
from src.data.database import DatabaseManager
from src.features.engineering import FeatureEngineer
from src.models.engagement_predictor import EngagementPredictor
from src.optimization.contextual_bandit import PostingTimeOptimizer
from src.scheduling.scheduler import PostingScheduler

def main():
    """Generate posting schedule recommendations."""
    parser = argparse.ArgumentParser(description='Generate posting schedule')
    parser.add_argument('--channel_id', required=True, help='YouTube channel ID')
    parser.add_argument('--days_ahead', type=int, default=7, help='Days to schedule ahead')
    parser.add_argument('--model_path', default='models/engagement_predictor.pkl',
                       help='Path to trained model')
    parser.add_argument('--bandit_state_path', default='models/bandit_state.json',
                       help='Path to bandit state file')
    parser.add_argument('--output_path', default='outputs/schedule.json',
                       help='Path to save schedule')
    parser.add_argument('--create_visualizations', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        db_manager = DatabaseManager(config.get('database.url'))
        feature_engineer = FeatureEngineer(
            embedding_model=config.get('features.text_embedding.model')
        )
        
        # Load trained model
        logger.info(f"Loading trained model from {args.model_path}")
        predictor = EngagementPredictor()
        predictor.load_model(args.model_path)
        
        # Initialize bandit optimizer
        bandit_params = config.get('models.bandit.params')
        optimizer = PostingTimeOptimizer(
            bandit_algorithm=config.get('models.bandit.algorithm'),
            bandit_params=bandit_params
        )
        
        # Load bandit state if exists
        if os.path.exists(args.bandit_state_path):
            logger.info(f"Loading bandit state from {args.bandit_state_path}")
            optimizer.load_state(args.bandit_state_path)
        else:
            logger.info("No existing bandit state found, starting fresh")
        
        # Initialize scheduler
        scheduler = PostingScheduler(predictor, optimizer, feature_engineer)
        
        # Get recent channel performance for context
        logger.info("Gathering channel context")
        recent_df = db_manager.get_videos_dataframe(
            channel_id=args.channel_id,
            days_back=30
        )
        
        if recent_df.empty:
            logger.warning("No recent data found for context, using defaults")
            context = {
                'channel_id': args.channel_id,
                'recent_avg_views': 10000,
                'recent_avg_engagement': 0.05,
                'views_trend': 0.0
            }
        else:
            context = {
                'channel_id': args.channel_id,
                'recent_avg_views': recent_df['views'].mean(),
                'recent_avg_engagement': recent_df['engagement_rate'].mean(),
                'views_trend': recent_df['views'].pct_change().mean()
            }
        
        logger.info(f"Channel context: {context}")
        
        # Generate schedule
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date += timedelta(days=1)  # Start from tomorrow
        
        logger.info(f"Generating {args.days_ahead}-day schedule starting {start_date.strftime('%Y-%m-%d')}")
        
        if args.days_ahead == 1:
            # Single day schedule
            daily_schedule = scheduler.generate_daily_schedule(start_date, context)
            schedule = {start_date.strftime('%Y-%m-%d'): daily_schedule}
        else:
            # Multi-day schedule
            schedule = scheduler.generate_weekly_schedule(
                start_date, context, 
                top_k_per_day=config.get('scheduling.top_k_recommendations')
            )
        
        # Save schedule
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        scheduler.export_schedule(schedule, args.output_path, format='json')
        
        # Also save as CSV
        csv_path = args.output_path.replace('.json', '.csv')
        scheduler.export_schedule(schedule, csv_path, format='csv')
        
        # Generate summary
        summary = scheduler.get_schedule_summary(schedule)
        logger.info("Schedule generation completed")
        logger.info(f"Total recommendations: {summary['total_recommendations']}")
        logger.info(f"Average expected views: {summary['avg_expected_views']:,.0f}")
        logger.info(f"Average confidence: {summary['avg_confidence']:.3f}")
        
        # Log top recommended hours
        logger.info("Most recommended hours:")
        for hour, count in list(summary['most_recommended_hours'].items())[:5]:
            logger.info(f"  {hour:02d}:00 - {count} times")
        
        # Create visualizations if requested
        if args.create_visualizations:
            logger.info("Creating visualizations")
            
            # Heatmap
            heatmap_path = args.output_path.replace('.json', '_heatmap.html')
            heatmap_fig = scheduler.create_heatmap_visualization(context, heatmap_path)
            
            # Performance dashboard (if we have historical data)
            if not recent_df.empty:
                dashboard_path = args.output_path.replace('.json', '_dashboard.html')
                dashboard_fig = scheduler.create_performance_dashboard(recent_df, dashboard_path)
            
            logger.info("Visualizations created")
        
        # Save updated bandit state
        optimizer.save_state(args.bandit_state_path)
        
        # Print schedule summary to console
        print("\n" + "="*60)
        print("POSTING SCHEDULE RECOMMENDATIONS")
        print("="*60)
        
        for date_str, recommendations in schedule.items():
            print(f"\n📅 {date_str}")
            print("-" * 40)
            
            for rec in recommendations:
                confidence_bar = "█" * int(rec['combined_confidence'] * 20)
                print(f"  {rec['rank']}. {rec['time_display']} "
                      f"({rec['expected_views']:,} views) "
                      f"[{confidence_bar:<20}] {rec['combined_confidence']:.3f}")
        
        print(f"\n📊 Schedule saved to: {args.output_path}")
        if args.create_visualizations:
            print(f"📈 Visualizations saved to: {os.path.dirname(args.output_path)}")
        
    except Exception as e:
        logger.error(f"Schedule generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()