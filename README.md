# PulseSlot AI - YouTube Posting Time Optimizer

AI-driven system that optimizes YouTube Shorts posting times using engagement prediction and contextual bandit optimization.

## Features

- **Engagement Prediction**: XGBoost model with uncertainty estimation
- **Adaptive Learning**: Thompson Sampling contextual bandit
- **Multi-Country Training**: 370K+ videos from 10 countries
- **Memory Efficient**: Processes datasets sequentially
- **No API Required**: Train immediately with included datasets

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test setup (30 seconds)
python test_dataset.py

# 3. Train model (2-5 minutes)
python main.py train --countries US --sample_size 5000

# 4. Generate schedule
python main.py schedule --channel_id YOUR_CHANNEL
```

## Dataset

Includes YouTube trending videos from 10 countries in the `data/` directory:
- **US, GB, CA, DE, FR, IN, JP, KR, MX, RU**
- **419K+ total videos**
- **~514MB total size**

## Training

### Quick Test
```bash
python main.py train --countries US --sample_size 5000
```

### Single Country
```bash
python main.py train --train_countries US
```

### Multi-Country (Recommended)
```bash
python main.py train \
  --train_countries US GB \
  --finetune_countries CA DE FR
```

## Commands

```bash
# Main CLI
python main.py test              # Verify setup
python main.py explore --country US  # Explore data
python main.py train --countries US  # Train model
python main.py schedule --channel_id ID  # Generate schedule
python main.py init-db           # Initialize database

# Direct scripts
python scripts/explore_dataset.py --country US
python scripts/train_model.py --countries US --sample_size 5000
python scripts/generate_schedule.py --channel_id YOUR_CHANNEL
```

## Architecture

```
PulseSlot-ai/
├── README.md                 # Main documentation (4.4KB)
├── USAGE.md                  # Quick reference (7.5KB)
├── instructions.md           # Original instructions (4.2KB)
├── OPTIMIZATION_SUMMARY.md   # Optimization details (4.3KB)
│
├── main.py                   # CLI interface
├── test_dataset.py           # Quick verification
│
├── src/                      # Core implementation (8 modules)
│   ├── data/
│   │   ├── dataset_loader.py      # Dataset loading
│   │   ├── database.py            # Database operations
│   │   └── youtube_api.py         # YouTube API client
│   ├── features/
│   │   └── engineering.py         # Feature engineering
│   ├── models/
│   │   └── engagement_predictor.py # ML model
│   ├── optimization/
│   │   └── contextual_bandit.py   # Bandit algorithms
│   ├── scheduling/
│   │   └── scheduler.py           # Schedule generation
│   └── utils/
│       └── config.py              # Configuration (optimized)
│
├── scripts/                  # Executable scripts (5 files)
│   ├── explore_dataset.py
│   ├── train_model.py
│   ├── generate_schedule.py
│   ├── init_db.py
│   └── collect_data.py
│
├── data/                     # 419K+ videos, 10 countries
├── config/                   # Configuration files
├── requirements.txt          # Dependencies
└── pyproject.toml           # Project metadata
```

## Configuration

Edit `config/config.yaml`:

```yaml
models:
  engagement_predictor:
    params:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1

  bandit:
    algorithm: "thompson_sampling"
    params:
      alpha: 1.0
      beta: 1.0

scheduling:
  top_k_recommendations: 3
  confidence_threshold: 0.7
```

## Memory Optimization

The system processes one country at a time to minimize RAM usage:

- **Sample mode (5K videos)**: 2-3 GB RAM
- **Single country (40K)**: 4-6 GB RAM
- **Multi-country**: 4-6 GB RAM (peak)

## Troubleshooting

**Out of Memory**
```bash
python main.py train --countries US --sample_size 3000
```

**Missing Dependencies**
```bash
pip install -r requirements.txt --upgrade
```

**Dataset Not Found**
```bash
ls -lh dataset/  # Verify files exist
```

## Advanced Usage

### Custom Training
```python
from src.data.dataset_loader import DatasetLoader
from src.features.engineering import FeatureEngineer
from src.models.engagement_predictor import EngagementPredictor

loader = DatasetLoader()
df = loader.load_country_data('US')
df = loader.preprocess_dataframe(df, loader.load_category_mapping('US'))

engineer = FeatureEngineer()
df_features, embeddings = engineer.fit_transform(df, 'US')

predictor = EngagementPredictor()
metrics = predictor.fit(df_features, embeddings)
predictor.save_model('models/custom_model.pkl')
```

### Database Integration
```bash
# Load datasets into database
python main.py init-db --load-dataset --countries US GB

# Train from database
python scripts/train_model.py --use-database
```

## Performance

- **Training Time**: 2-5 min (sample), 10-30 min (full)
- **Prediction**: <100ms per video
- **Schedule Generation**: <1 second

## Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- 1GB disk space

## License

See LICENSE file for details.

## Support

For issues or questions, see `instructions.md` or check the code documentation.
