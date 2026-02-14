#!/usr/bin/env python3
"""Train engagement prediction model."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
from datetime import datetime, timedelta
import gc
import pandas as pd

from src.utils.config import Config, setup_logging
from src.data.dataset_loader import DatasetLoader
from src.features.engineering import FeatureEngineer
from src.models.engagement_predictor import EngagementPredictor

def main():
    """Train engagement prediction model using dataset files."""
    parser = argparse.ArgumentParser(description='Train engagement prediction model')
    parser.add_argument('--countries', nargs='+', 
                       help='Country codes to use (e.g., US GB CA). If not specified, uses all')
    parser.add_argument('--train_countries', nargs='+',
                       help='Countries for initial training (default: first country)')
    parser.add_argument('--finetune_countries', nargs='+',
                       help='Countries for fine-tuning (default: remaining countries)')
    parser.add_argument('--sample_size', type=int, 
                       help='Sample size per country (for testing, uses all if not specified)')
    parser.add_argument('--model_path', default='models/engagement_predictor.pkl', 
                       help='Path to save trained model')
    parser.add_argument('--features_path', default='models/feature_engineer.pkl',
                       help='Path to save feature engineer')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize components
    dataset_loader = DatasetLoader()
    feature_engineer = FeatureEngineer(embedding_model="all-MiniLM-L6-v2")
    predictor = EngagementPredictor()
    
    try:
        # Determine which countries to use
        available_countries = dataset_loader.available_countries
        
        if args.countries:
            countries = [c for c in args.countries if c in available_countries]
        else:
            countries = available_countries
        
        if not countries:
            logger.error("No valid countries specified")
            sys.exit(1)
        
        logger.info(f"Using countries: {', '.join(countries)}")
        
        # Split into training and fine-tuning sets
        if args.train_countries:
            train_countries = [c for c in args.train_countries if c in countries]
        else:
            train_countries = [countries[0]]  # Use first country for training
        
        if args.finetune_countries:
            finetune_countries = [c for c in args.finetune_countries if c in countries]
        else:
            finetune_countries = [c for c in countries if c not in train_countries]
        
        logger.info(f"Training countries: {', '.join(train_countries)}")
        logger.info(f"Fine-tuning countries: {', '.join(finetune_countries)}")
        
        # Phase 1: Initial training on first country/countries
        logger.info("=" * 60)
        logger.info("PHASE 1: Initial Model Training")
        logger.info("=" * 60)
        
        all_train_data = []
        for country in train_countries:
            logger.info(f"Loading training data from {country}")
            df = dataset_loader.load_country_data(country)
            category_map = dataset_loader.load_category_mapping(country)
            df = dataset_loader.preprocess_dataframe(df, category_map)
            
            # Sample if requested
            if args.sample_size and len(df) > args.sample_size:
                df = df.sample(n=args.sample_size, random_state=42)
                logger.info(f"Sampled {args.sample_size} videos from {country}")
            
            # Add country identifier
            df['country'] = country
            all_train_data.append(df)
            
            logger.info(f"Loaded {len(df)} videos from {country}")
        
        # Combine training data
        df_train = pd.concat(all_train_data, ignore_index=True)
        logger.info(f"Total training samples: {len(df_train)}")
        
        # Filter outliers
        min_views = df_train['views'].quantile(0.05)
        max_views = df_train['views'].quantile(0.95)
        df_train = df_train[(df_train['views'] >= min_views) & (df_train['views'] <= max_views)]
        logger.info(f"Filtered to {len(df_train)} videos (views: {min_views:.0f} - {max_views:.0f})")
        
        # Feature engineering
        logger.info("Performing feature engineering on training data")
        df_features, embeddings = feature_engineer.fit_transform(df_train, 'multi_country')
        
        # Train initial model
        logger.info("Training initial engagement prediction model")
        training_metrics = predictor.fit(df_features, embeddings, n_bootstrap=10)
        
        logger.info("Initial training completed")
        logger.info(f"Training RMSE: {training_metrics['rmse']:.4f}")
        logger.info(f"Training R²: {training_metrics['r2']:.4f}")
        logger.info(f"CV RMSE: {training_metrics['cv_rmse_mean']:.4f} ± {training_metrics['cv_rmse_std']:.4f}")
        
        # Clean up training data
        del df_train, all_train_data, df_features, embeddings
        gc.collect()
        
        # Phase 2: Fine-tuning on additional countries
        if finetune_countries:
            logger.info("=" * 60)
            logger.info("PHASE 2: Model Fine-tuning")
            logger.info("=" * 60)
            
            for country in finetune_countries:
                logger.info(f"Fine-tuning on {country} data")
                
                df = dataset_loader.load_country_data(country)
                category_map = dataset_loader.load_category_mapping(country)
                df = dataset_loader.preprocess_dataframe(df, category_map)
                
                # Sample if requested
                if args.sample_size and len(df) > args.sample_size:
                    df = df.sample(n=args.sample_size, random_state=42)
                
                df['country'] = country
                logger.info(f"Loaded {len(df)} videos from {country}")
                
                # Filter outliers
                min_views = df['views'].quantile(0.05)
                max_views = df['views'].quantile(0.95)
                df = df[(df['views'] >= min_views) & (df['views'] <= max_views)]
                
                # Transform features (using already fitted feature engineer)
                logger.info("Transforming features")
                df_features, embeddings = feature_engineer.transform(df, country)
                
                # Fine-tune model (continue training)
                logger.info(f"Fine-tuning model on {country}")
                finetune_metrics = predictor.fit(df_features, embeddings, n_bootstrap=5)
                
                logger.info(f"Fine-tuning on {country} completed")
                logger.info(f"RMSE: {finetune_metrics['rmse']:.4f}, R²: {finetune_metrics['r2']:.4f}")
                
                # Clean up
                del df, df_features, embeddings
                gc.collect()
        
        # Final evaluation
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 60)
        
        # Evaluate on a test country (last one)
        test_country = countries[-1]
        logger.info(f"Evaluating on {test_country}")
        
        df_test = dataset_loader.load_country_data(test_country)
        category_map = dataset_loader.load_category_mapping(test_country)
        df_test = dataset_loader.preprocess_dataframe(df_test, category_map)
        df_test['country'] = test_country
        
        # Sample for evaluation
        if len(df_test) > 5000:
            df_test = df_test.sample(n=5000, random_state=42)
        
        df_test_features, test_embeddings = feature_engineer.transform(df_test, test_country)
        eval_metrics = predictor.evaluate_time_series(df_test_features, test_embeddings)
        
        logger.info("Final evaluation metrics:")
        logger.info(f"Test RMSE: {eval_metrics['test_rmse']:.0f}")
        logger.info(f"Test MAE: {eval_metrics['test_mae']:.0f}")
        logger.info(f"Test R²: {eval_metrics['test_r2']:.4f}")
        logger.info(f"Test MAPE: {eval_metrics['test_mape']:.2f}%")
        
        # Feature importance
        feature_importance = predictor.get_feature_importance(top_k=15)
        logger.info("Top 15 most important features:")
        for _, row in feature_importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save models
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        predictor.save_model(args.model_path)
        feature_engineer.save(args.features_path)
        
        logger.info(f"Models saved successfully")
        
        # Generate training summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'train_countries': train_countries,
            'finetune_countries': finetune_countries,
            'test_country': test_country,
            'training_metrics': training_metrics,
            'evaluation_metrics': eval_metrics,
            'model_path': args.model_path,
            'features_path': args.features_path
        }
        
        import json
        summary_path = args.model_path.replace('.pkl', '_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training summary saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()