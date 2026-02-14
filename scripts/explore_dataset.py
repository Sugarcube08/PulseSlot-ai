#!/usr/bin/env python3
"""Explore the YouTube trending videos dataset."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
from src.data.dataset_loader import DatasetLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Explore dataset statistics and structure."""
    parser = argparse.ArgumentParser(description='Explore YouTube dataset')
    parser.add_argument('--country', help='Show detailed stats for specific country')
    parser.add_argument('--sample', type=int, help='Show sample data (N rows)')
    
    args = parser.parse_args()
    
    # Initialize dataset loader
    loader = DatasetLoader()
    
    # Show overall statistics
    logger.info("=" * 60)
    logger.info("DATASET OVERVIEW")
    logger.info("=" * 60)
    
    stats_df = loader.get_dataset_stats()
    print("\nAvailable Datasets:")
    print(stats_df.to_string(index=False))
    print(f"\nTotal videos: {stats_df['videos'].sum():,}")
    print(f"Total size: {stats_df['file_size_mb'].sum():.2f} MB")
    
    # Country-specific analysis
    if args.country:
        logger.info("=" * 60)
        logger.info(f"DETAILED ANALYSIS: {args.country}")
        logger.info("=" * 60)
        
        try:
            # Load data
            df = loader.load_country_data(args.country)
            category_map = loader.load_category_mapping(args.country)
            df = loader.preprocess_dataframe(df, category_map)
            
            print(f"\nTotal videos: {len(df):,}")
            print(f"Date range: {df['published_at'].min()} to {df['published_at'].max()}")
            print(f"\nUnique channels: {df['channel_title'].nunique():,}")
            
            # View statistics
            print("\nView Statistics:")
            print(f"  Mean: {df['views'].mean():,.0f}")
            print(f"  Median: {df['views'].median():,.0f}")
            print(f"  Min: {df['views'].min():,}")
            print(f"  Max: {df['views'].max():,}")
            
            # Category distribution
            print("\nTop 10 Categories:")
            category_counts = df['category_name'].value_counts().head(10)
            for cat, count in category_counts.items():
                print(f"  {cat}: {count:,} ({count/len(df)*100:.1f}%)")
            
            # Time distribution
            print("\nPosting Hour Distribution:")
            hour_dist = df['hour_of_day'].value_counts().sort_index()
            for hour, count in hour_dist.items():
                bar = '█' * int(count / hour_dist.max() * 50)
                print(f"  {hour:02d}:00 | {bar} {count:,}")
            
            # Day of week distribution
            print("\nDay of Week Distribution:")
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_dist = df['day_of_week'].value_counts().sort_index()
            for dow, count in dow_dist.items():
                bar = '█' * int(count / dow_dist.max() * 50)
                print(f"  {days[dow]}: {bar} {count:,}")
            
            # Engagement statistics
            print("\nEngagement Metrics:")
            print(f"  Avg like rate: {df['like_rate'].mean():.4f}")
            print(f"  Avg comment rate: {df['comment_rate'].mean():.4f}")
            print(f"  Avg engagement rate: {df['engagement_rate'].mean():.4f}")
            
            # Top channels
            print("\nTop 10 Channels by Video Count:")
            top_channels = df['channel_title'].value_counts().head(10)
            for channel, count in top_channels.items():
                avg_views = df[df['channel_title'] == channel]['views'].mean()
                print(f"  {channel}: {count} videos (avg {avg_views:,.0f} views)")
            
            # Sample data
            if args.sample:
                print(f"\nSample Data ({args.sample} rows):")
                sample_cols = ['title', 'channel_title', 'category_name', 'views', 
                             'likes', 'comments', 'published_at']
                print(df[sample_cols].head(args.sample).to_string(index=False))
            
        except Exception as e:
            logger.error(f"Error analyzing {args.country}: {e}")
            import traceback
            traceback.print_exc()
    
    # Show recommendations
    logger.info("=" * 60)
    logger.info("TRAINING RECOMMENDATIONS")
    logger.info("=" * 60)
    
    print("\nRecommended training strategies:")
    print("1. Quick test (sample data):")
    print("   python scripts/train_model.py --countries US --sample_size 5000")
    print("\n2. Single country training:")
    print("   python scripts/train_model.py --train_countries US")
    print("\n3. Multi-country training with fine-tuning:")
    print("   python scripts/train_model.py --train_countries US GB --finetune_countries CA DE FR")
    print("\n4. All countries (memory intensive):")
    print("   python scripts/train_model.py")


if __name__ == "__main__":
    main()
