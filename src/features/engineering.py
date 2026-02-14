"""Feature engineering for engagement prediction and contextual bandits."""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import re

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline for YouTube engagement prediction."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize feature engineer.
        
        Args:
            embedding_model: Sentence transformer model name
        """
        self.embedding_model_name = embedding_model
        self.sentence_transformer = None
        self.category_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.is_fitted = False
        
    def _load_embedding_model(self):
        """Lazy load sentence transformer model."""
        if self.sentence_transformer is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.sentence_transformer = SentenceTransformer(self.embedding_model_name)
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from published_at timestamp.
        
        Args:
            df: DataFrame with published_at column
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['published_at']):
            df['published_at'] = pd.to_datetime(df['published_at'])
        
        # Extract time features
        df['hour_of_day'] = df['published_at'].dt.hour
        df['day_of_week'] = df['published_at'].dt.dayofweek  # 0=Monday
        df['day_of_month'] = df['published_at'].dt.day
        df['month'] = df['published_at'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time of day categories
        df['time_category'] = pd.cut(
            df['hour_of_day'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        return df
    
    def extract_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract engagement-based features.
        
        Args:
            df: DataFrame with views, likes, comments columns
            
        Returns:
            DataFrame with engagement features
        """
        df = df.copy()
        
        # Engagement rates (handle division by zero)
        df['like_rate'] = df['likes'] / (df['views'] + 1)
        df['comment_rate'] = df['comments'] / (df['views'] + 1)
        df['engagement_rate'] = (df['likes'] + df['comments']) / (df['views'] + 1)
        
        # Log transformations for skewed distributions
        df['log_views'] = np.log1p(df['views'])
        df['log_likes'] = np.log1p(df['likes'])
        df['log_comments'] = np.log1p(df['comments'])
        
        # Engagement velocity (if we have time-series data)
        if 'published_at' in df.columns:
            df = df.sort_values('published_at')
            df['views_per_hour'] = df['views'] / (
                (datetime.utcnow() - df['published_at']).dt.total_seconds() / 3600 + 1
            )
        
        return df
    
    def extract_text_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Extract text-based features from titles and descriptions.
        
        Args:
            df: DataFrame with title and description columns
            fit: Whether to fit the text processors
            
        Returns:
            DataFrame with text features
        """
        df = df.copy()
        
        # Basic text statistics
        df['title_length'] = df['title'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        df['has_description'] = (df['description'].str.len() > 0).astype(int)
        df['description_length'] = df['description'].str.len()
        
        # Title sentiment and characteristics
        df['title_has_question'] = df['title'].str.contains(r'\?', regex=True).astype(int)
        df['title_has_exclamation'] = df['title'].str.contains(r'!', regex=True).astype(int)
        df['title_has_numbers'] = df['title'].str.contains(r'\d', regex=True).astype(int)
        df['title_all_caps_words'] = df['title'].str.count(r'\b[A-Z]{2,}\b')
        
        # Common YouTube keywords
        clickbait_words = ['amazing', 'incredible', 'shocking', 'unbelievable', 'secret', 
                          'hack', 'trick', 'viral', 'must', 'watch', 'epic']
        df['clickbait_score'] = df['title'].str.lower().apply(
            lambda x: sum(word in x for word in clickbait_words)
        )
        
        # TF-IDF features for titles
        if fit:
            title_tfidf = self.tfidf_vectorizer.fit_transform(df['title'].fillna(''))
        else:
            title_tfidf = self.tfidf_vectorizer.transform(df['title'].fillna(''))
            
        # Add TF-IDF features to dataframe
        tfidf_feature_names = [f'title_tfidf_{i}' for i in range(title_tfidf.shape[1])]
        tfidf_df = pd.DataFrame(
            title_tfidf.toarray(),
            columns=tfidf_feature_names,
            index=df.index
        )
        df = pd.concat([df, tfidf_df], axis=1)
        
        return df
    
    def extract_title_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """Extract sentence embeddings from video titles.
        
        Args:
            df: DataFrame with title column
            
        Returns:
            Array of title embeddings
        """
        self._load_embedding_model()
        
        titles = df['title'].fillna('').tolist()
        embeddings = self.sentence_transformer.encode(titles, show_progress_bar=True)
        
        return embeddings
    
    def extract_tag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from video tags.
        
        Args:
            df: DataFrame with tags column (list of strings)
            
        Returns:
            DataFrame with tag features
        """
        df = df.copy()
        
        # Handle missing tags
        df['tags'] = df['tags'].fillna('').apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        # Tag statistics
        df['tag_count'] = df['tags'].apply(len)
        df['avg_tag_length'] = df['tags'].apply(
            lambda tags: np.mean([len(tag) for tag in tags]) if tags else 0
        )
        
        # Common tag categories (you can expand this based on your domain)
        gaming_tags = ['gaming', 'game', 'gameplay', 'gamer', 'play']
        music_tags = ['music', 'song', 'audio', 'sound', 'beat']
        tutorial_tags = ['tutorial', 'how', 'guide', 'learn', 'tips']
        
        df['has_gaming_tags'] = df['tags'].apply(
            lambda tags: any(any(gt in tag.lower() for gt in gaming_tags) for tag in tags)
        ).astype(int)
        
        df['has_music_tags'] = df['tags'].apply(
            lambda tags: any(any(mt in tag.lower() for mt in music_tags) for tag in tags)
        ).astype(int)
        
        df['has_tutorial_tags'] = df['tags'].apply(
            lambda tags: any(any(tt in tag.lower() for tt in tutorial_tags) for tag in tags)
        ).astype(int)
        
        return df
    
    def create_context_features(self, df: pd.DataFrame, channel_id: str) -> pd.DataFrame:
        """Create contextual features for bandit algorithm.
        
        Args:
            df: DataFrame with video data
            channel_id: Channel identifier
            
        Returns:
            DataFrame with context features
        """
        df = df.copy()
        
        # Channel-specific features
        df['channel_id'] = channel_id
        
        # Recent performance context (rolling averages)
        df = df.sort_values('published_at')
        
        # 7-day rolling averages
        df['recent_avg_views'] = df['views'].rolling(window=7, min_periods=1).mean()
        df['recent_avg_engagement'] = df['engagement_rate'].rolling(window=7, min_periods=1).mean()
        
        # Performance trend
        df['views_trend'] = df['views'].pct_change(periods=3).fillna(0)
        
        # Day of week performance
        dow_performance = df.groupby('day_of_week')['views'].mean()
        df['dow_avg_performance'] = df['day_of_week'].map(dow_performance)
        
        # Hour of day performance
        hour_performance = df.groupby('hour_of_day')['views'].mean()
        df['hour_avg_performance'] = df['hour_of_day'].map(hour_performance)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, channel_id: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Fit feature engineering pipeline and transform data.
        
        Args:
            df: Raw video data DataFrame
            channel_id: Channel identifier
            
        Returns:
            Tuple of (transformed DataFrame, title embeddings)
        """
        logger.info("Fitting feature engineering pipeline")
        
        # Extract all features
        df = self.extract_time_features(df)
        df = self.extract_engagement_features(df)
        df = self.extract_text_features(df, fit=True)
        df = self.extract_tag_features(df)
        df = self.create_context_features(df, channel_id)
        
        # Get title embeddings
        embeddings = self.extract_title_embeddings(df)
        
        # Encode categorical features
        if 'category_id' in df.columns:
            df['category_encoded'] = self.category_encoder.fit_transform(df['category_id'].fillna(0))
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        self.is_fitted = True
        logger.info("Feature engineering pipeline fitted successfully")
        
        return df, embeddings
    
    def transform(self, df: pd.DataFrame, channel_id: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Transform new data using fitted pipeline.
        
        Args:
            df: Raw video data DataFrame
            channel_id: Channel identifier
            
        Returns:
            Tuple of (transformed DataFrame, title embeddings)
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
            
        # Extract all features
        df = self.extract_time_features(df)
        df = self.extract_engagement_features(df)
        df = self.extract_text_features(df, fit=False)
        df = self.extract_tag_features(df)
        df = self.create_context_features(df, channel_id)
        
        # Get title embeddings
        embeddings = self.extract_title_embeddings(df)
        
        # Transform categorical features
        if 'category_id' in df.columns:
            df['category_encoded'] = self.category_encoder.transform(df['category_id'].fillna(0))
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df, embeddings
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after transformation.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
            
        # This would need to be implemented based on the actual features created
        # For now, return a placeholder
        return ['feature_names_would_be_here']