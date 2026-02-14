#!/usr/bin/env python3
"""Collect YouTube data for a channel."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
from datetime import datetime

from src.utils.config import Config, setup_logging
from src.data.youtube_api import YouTubeAPIClient
from src.data.database import DatabaseManager
from src.features.engineering import FeatureEngineer

def main():
    """Collect YouTube data for analysis."""
    parser = argparse.ArgumentParser(description='Collect YouTube data')
    parser.add_argument('--channel_id', required=True, help='YouTube channel ID')
    parser.add_argument('--max_videos', type=int, default=100, help='Maximum videos to collect')
    parser.add_argument('--update_existing', action='store_true', help='Update existing videos')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Initialize components
    api_key = config.get('youtube.api_key')
    if not api_key:
        logger.error("YouTube API key not found in configuration")
        sys.exit(1)
    
    youtube_client = YouTubeAPIClient(api_key)
    db_manager = DatabaseManager(config.get('database.url'))
    feature_engineer = FeatureEngineer()
    
    try:
        logger.info(f"Collecting data for channel: {args.channel_id}")
        
        # Collect video data
        videos_data = youtube_client.get_channel_videos(
            args.channel_id, 
            max_results=args.max_videos
        )
        
        if not videos_data:
            logger.warning("No videos found for the channel")
            return
        
        logger.info(f"Collected {len(videos_data)} videos")
        
        # Process and enhance data
        for video in videos_data:
            # Extract time features
            published_at = video['published_at']
            video['hour_of_day'] = published_at.hour
            video['day_of_week'] = published_at.weekday()
            
            # Calculate engagement metrics
            views = video['views']
            likes = video['likes']
            comments = video['comments']
            
            if views > 0:
                video['engagement_rate'] = (likes + comments) / views
                video['like_rate'] = likes / views
            else:
                video['engagement_rate'] = 0.0
                video['like_rate'] = 0.0
        
        # Store in database
        stored_count = db_manager.store_videos(videos_data)
        logger.info(f"Stored {stored_count} videos in database")
        
        # Generate summary statistics
        total_views = sum(v['views'] for v in videos_data)
        avg_views = total_views / len(videos_data) if videos_data else 0
        
        logger.info(f"Data collection summary:")
        logger.info(f"  Total videos: {len(videos_data)}")
        logger.info(f"  Total views: {total_views:,}")
        logger.info(f"  Average views: {avg_views:,.0f}")
        
        # Hour distribution
        hour_dist = {}
        for video in videos_data:
            hour = video['hour_of_day']
            hour_dist[hour] = hour_dist.get(hour, 0) + 1
        
        logger.info("Posting hour distribution:")
        for hour in sorted(hour_dist.keys()):
            logger.info(f"  {hour:02d}:00 - {hour_dist[hour]} videos")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()