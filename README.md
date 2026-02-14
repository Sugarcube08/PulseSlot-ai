# Adaptive AI Posting Intelligence Engine for YouTube Shorts

An AI-driven adaptive scheduling system that optimizes YouTube Shorts posting times using engagement prediction and contextual bandit optimization.

## Features

- Engagement prediction using XGBoost
- Contextual bandit optimization (Thompson Sampling)
- Real-time learning from posting performance
- Confidence-scored recommendations
- Weekly heatmap visualizations
- Modular, scalable architecture

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up YouTube API credentials
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your YouTube API key

# Initialize database
python scripts/init_db.py

# Collect initial data
python scripts/collect_data.py --channel_id YOUR_CHANNEL_ID

# Train initial model
python scripts/train_model.py

# Generate recommendations
python scripts/generate_schedule.py
```

## Architecture

- **Data Layer**: YouTube API integration and storage
- **Feature Engineering**: Time features, embeddings, engagement ratios
- **Prediction Model**: XGBoost for engagement forecasting
- **Optimization**: Contextual bandit for time slot selection
- **Scheduling**: Daily/weekly recommendation engine
- **Learning Loop**: Continuous model updates

## Project Structure

```
src/
├── data/           # Data collection and storage
├── features/       # Feature engineering
├── models/         # ML models and training
├── optimization/   # Contextual bandit algorithms
├── scheduling/     # Recommendation engine
└── utils/          # Utilities and helpers
```