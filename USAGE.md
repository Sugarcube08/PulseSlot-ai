# Usage Guide

Quick reference for common tasks.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Test
python test_dataset.py

# 2. Explore
python main.py explore --country US

# 3. Train
python main.py train --countries US --sample_size 5000

# 4. Schedule
python main.py schedule --channel_id YOUR_CHANNEL
```

## Training Options

### Sample Training (Fast)
```bash
python main.py train --countries US --sample_size 5000
```

### Single Country
```bash
python main.py train --train_countries US
```

### Multi-Country
```bash
python main.py train \
  --train_countries US GB \
  --finetune_countries CA DE FR
```

### All Countries
```bash
python main.py train
```

## Data Exploration

```bash
# Overview
python main.py explore

# Specific country
python main.py explore --country US --sample 10

# Direct script
python scripts/explore_dataset.py --country US
```

## Schedule Generation

```bash
# Generate schedule
python main.py schedule --channel_id YOUR_CHANNEL

# With visualizations
python scripts/generate_schedule.py \
  --channel_id YOUR_CHANNEL \
  --days_ahead 7 \
  --create_visualizations
```

## Database Operations

```bash
# Initialize
python main.py init-db

# Load datasets
python main.py init-db --load-dataset --countries US GB

# Collect from API
python scripts/collect_data.py --channel_id YOUR_CHANNEL
```

## Common Workflows

### First Time Setup
```bash
python test_dataset.py
python main.py explore --country US
python main.py train --countries US --sample_size 5000
cat models/engagement_predictor_summary.json
```

### Production Training
```bash
python main.py train --train_countries US GB --finetune_countries CA DE FR
python main.py schedule --channel_id YOUR_CHANNEL
```

### Development Iteration
```bash
# Quick test
python main.py train --countries US --sample_size 3000

# Make changes...

# Test again
python main.py train --countries US --sample_size 5000
```

## Troubleshooting

### Out of Memory
```bash
# Reduce sample size
python main.py train --countries US --sample_size 2000

# Train on one country
python main.py train --train_countries US
```

### Slow Training
```bash
# Use smaller sample
python main.py train --countries US --sample_size 3000

# Train on fewer countries
python main.py train --train_countries US
```

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

## Environment Variables

```bash
# Custom dataset directory
export DATASET_DIR=/path/to/datasets

# Custom model directory
export MODEL_DIR=/path/to/models

# Debug logging
export LOG_LEVEL=DEBUG
```

## Performance Tips

1. **Start with samples** - Always test with `--sample_size 5000` first
2. **Monitor memory** - Use `htop` or Task Manager
3. **Close apps** - Free up RAM before training
4. **GPU** - XGBoost will use GPU if available

## File Locations

```
data/                       # Dataset files (419K+ videos)
models/                     # Trained models
├── engagement_predictor.pkl
├── engagement_predictor_summary.json
└── feature_engineer.pkl

outputs/                    # Generated schedules
├── schedule.json
├── schedule.csv
└── schedule_heatmap.html

logs/                       # Application logs
config/                     # Configuration
```

## Help Commands

```bash
python main.py --help
python main.py train --help
python main.py explore --help
python scripts/train_model.py --help
python scripts/explore_dataset.py --help
```

## Configuration

Edit `config/config.yaml`:

```yaml
models:
  engagement_predictor:
    params:
      n_estimators: 100      # More = better but slower
      max_depth: 6           # Deeper = more complex
      learning_rate: 0.1     # Lower = more careful

scheduling:
  top_k_recommendations: 3   # Number of time slots
  confidence_threshold: 0.7  # Minimum confidence

features:
  text_embedding:
    model: "all-MiniLM-L6-v2"  # Embedding model
```

## API Reference

### DatasetLoader
```python
from src.data.dataset_loader import DatasetLoader

loader = DatasetLoader()
df = loader.load_country_data('US')
category_map = loader.load_category_mapping('US')
df = loader.preprocess_dataframe(df, category_map)
```

### FeatureEngineer
```python
from src.features.engineering import FeatureEngineer

engineer = FeatureEngineer()
df_features, embeddings = engineer.fit_transform(df, 'US')
engineer.save('models/feature_engineer.pkl')
```

### EngagementPredictor
```python
from src.models.engagement_predictor import EngagementPredictor

predictor = EngagementPredictor()
metrics = predictor.fit(df_features, embeddings)
predictions, uncertainties = predictor.predict(df_test, embeddings_test)
predictor.save_model('models/model.pkl')
```

### PostingTimeOptimizer
```python
from src.optimization.contextual_bandit import PostingTimeOptimizer

optimizer = PostingTimeOptimizer(bandit_algorithm="thompson_sampling")
recommendations = optimizer.recommend_posting_times(context, top_k=3)
optimizer.update_performance(context, hour, actual_views)
```

## Examples

### Train and Evaluate
```python
from src.data.dataset_loader import DatasetLoader
from src.features.engineering import FeatureEngineer
from src.models.engagement_predictor import EngagementPredictor

# Load data
loader = DatasetLoader()
df = loader.load_country_data('US')
df = loader.preprocess_dataframe(df, loader.load_category_mapping('US'))

# Feature engineering
engineer = FeatureEngineer()
df_features, embeddings = engineer.fit_transform(df, 'US')

# Train
predictor = EngagementPredictor()
metrics = predictor.fit(df_features, embeddings)

# Evaluate
eval_metrics = predictor.evaluate_time_series(df_features, embeddings)
print(f"Test R²: {eval_metrics['test_r2']:.3f}")
print(f"Test RMSE: {eval_metrics['test_rmse']:.0f}")
```

### Generate Schedule
```python
from src.scheduling.scheduler import PostingScheduler
from datetime import datetime

# Initialize scheduler (with loaded models)
scheduler = PostingScheduler(predictor, optimizer, engineer)

# Generate schedule
context = {'channel_id': 'YOUR_CHANNEL', 'recent_avg_views': 10000}
schedule = scheduler.generate_weekly_schedule(datetime.now(), context)

# Export
scheduler.export_schedule(schedule, 'outputs/schedule.json')
```

## Dataset Information

### Available Countries
- US: ~41K videos
- GB: ~44K videos  
- CA: ~46K videos
- DE: ~47K videos
- FR: ~46K videos
- IN: ~39K videos
- JP: ~22K videos
- KR: ~37K videos
- MX: ~44K videos
- RU: ~46K videos

### CSV Columns
- video_id, title, description
- channel_title, category_id
- publish_time, tags
- views, likes, dislikes, comment_count
- comments_disabled, ratings_disabled

### Category Mappings
JSON files map category IDs to names:
- 1: Film & Animation
- 10: Music
- 20: Gaming
- 22: People & Blogs
- 23: Comedy
- 24: Entertainment
- 25: News & Politics
- 26: Howto & Style
- 27: Education
- 28: Science & Technology

## Memory Usage

| Operation | RAM Usage |
|-----------|-----------|
| Sample (5K) | 2-3 GB |
| Single country (40K) | 4-6 GB |
| Multi-country | 4-6 GB (peak) |
| Prediction | <100 MB |

## Training Time

| Dataset | Time |
|---------|------|
| 5K sample | 2-5 min |
| 40K (1 country) | 10-15 min |
| 370K (all) | 30-45 min |

## Model Performance

Typical metrics on test data:
- R²: 0.70-0.80
- RMSE: Varies by scale
- MAE: Varies by scale
- MAPE: 15-25%

## Tips

1. Always start with `--sample_size 5000`
2. Monitor RAM with `htop` or Task Manager
3. Close unnecessary applications
4. Use GPU if available (automatic)
5. Save models frequently
6. Check logs for errors
7. Validate results before production use
