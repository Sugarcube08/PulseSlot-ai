#!/usr/bin/env python3
"""Quick test script to verify dataset loading."""

import logging
from src.data.dataset_loader import DatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test dataset loading functionality."""
    print("=" * 60)
    print("TESTING DATASET LOADER")
    print("=" * 60)
    
    # Initialize loader
    loader = DatasetLoader()
    
    # Show available datasets
    print("\n1. Available Datasets:")
    stats = loader.get_dataset_stats()
    print(stats.to_string(index=False))
    
    # Test loading a single country
    print("\n2. Testing Single Country Load (US):")
    try:
        df = loader.load_country_data('US')
        print(f"   ✓ Loaded {len(df):,} videos")
        print(f"   ✓ Columns: {', '.join(df.columns[:5])}...")
        
        # Test category mapping
        print("\n3. Testing Category Mapping:")
        category_map = loader.load_category_mapping('US')
        print(f"   ✓ Loaded {len(category_map)} categories")
        print(f"   ✓ Sample: {list(category_map.items())[:3]}")
        
        # Test preprocessing
        print("\n4. Testing Preprocessing:")
        df_processed = loader.preprocess_dataframe(df, category_map)
        print(f"   ✓ Processed {len(df_processed):,} videos")
        print(f"   ✓ New columns: hour_of_day, day_of_week, engagement_rate, etc.")
        print(f"   ✓ Date range: {df_processed['published_at'].min()} to {df_processed['published_at'].max()}")
        
        # Show sample
        print("\n5. Sample Data:")
        sample = df_processed[['title', 'views', 'likes', 'hour_of_day', 'category_name']].head(3)
        print(sample.to_string(index=False))
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python scripts/explore_dataset.py")
        print("  python scripts/train_model.py --countries US --sample_size 5000")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
