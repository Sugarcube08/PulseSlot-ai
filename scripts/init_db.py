#!/usr/bin/env python3
"""Initialize database tables and setup."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
from src.utils.config import Config, setup_logging, create_directories
from src.data.database import DatabaseManager
from src.data.dataset_loader import DatasetLoader


def main():
    """Initialize database and optionally load dataset."""
    parser = argparse.ArgumentParser(description='Initialize database')
    parser.add_argument('--load-dataset', action='store_true',
                       help='Load data from dataset files into database')
    parser.add_argument('--countries', nargs='+',
                       help='Country codes to load (default: all)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of videos per country')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Create directories
    create_directories()
    
    # Initialize database
    database_url = config.get('database.url')
    logger.info(f"Initializing database: {database_url}")
    
    db_manager = DatabaseManager(database_url)
    
    try:
        # Create all tables
        db_manager.create_tables()
        logger.info("Database tables created successfully")
        
        # Load dataset if requested
        if args.load_dataset:
            logger.info("=" * 60)
            logger.info("LOADING DATASET INTO DATABASE")
            logger.info("=" * 60)
            
            dataset_loader = DatasetLoader()
            countries = args.countries or dataset_loader.available_countries
            
            total_loaded = 0
            for country in countries:
                logger.info(f"Loading {country} data...")
                
                try:
                    df = dataset_loader.load_country_data(country)
                    category_map = dataset_loader.load_category_mapping(country)
                    df = dataset_loader.preprocess_dataframe(df, category_map)
                    
                    # Limit if requested
                    if args.limit and len(df) > args.limit:
                        df = df.sample(n=args.limit, random_state=42)
                    
                    # Convert to database format
                    videos_data = []
                    for _, row in df.iterrows():
                        video_dict = {
                            'video_id': row['video_id'],
                            'title': row['title'],
                            'description': row.get('description', ''),
                            'published_at': row['published_at'],
                            'channel_id': country,  # Using country as channel_id for now
                            'channel_title': row['channel_title'],
                            'category_id': int(row['category_id']),
                            'tags': row['tags'],
                            'views': int(row['views']),
                            'likes': int(row['likes']),
                            'comments': int(row['comments']),
                            'hour_of_day': int(row['hour_of_day']),
                            'day_of_week': int(row['day_of_week']),
                            'engagement_rate': float(row['engagement_rate']),
                            'like_rate': float(row['like_rate'])
                        }
                        videos_data.append(video_dict)
                    
                    # Store in database
                    stored = db_manager.store_videos(videos_data)
                    total_loaded += stored
                    logger.info(f"Loaded {stored} videos from {country}")
                    
                    # Clean up
                    del df, videos_data
                    
                except Exception as e:
                    logger.error(f"Error loading {country}: {e}")
                    continue
            
            logger.info(f"Total videos loaded: {total_loaded:,}")
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()