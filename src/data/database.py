"""Database models and operations for storing YouTube analytics data."""

import logging
from datetime import datetime
from typing import List, Dict, Optional
import json

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd

logger = logging.getLogger(__name__)

Base = declarative_base()


class Video(Base):
    """Video metadata and engagement metrics."""
    
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String(20), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    published_at = Column(DateTime, nullable=False, index=True)
    channel_id = Column(String(50), nullable=False, index=True)
    channel_title = Column(String(200))
    category_id = Column(Integer)
    tags = Column(JSON)
    duration = Column(String(20))
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    collected_at = Column(DateTime, default=datetime.utcnow)
    
    # Derived features
    hour_of_day = Column(Integer, index=True)
    day_of_week = Column(Integer, index=True)
    engagement_rate = Column(Float)
    like_rate = Column(Float)


class PerformanceHistory(Base):
    """Historical performance tracking at different time intervals."""
    
    __tablename__ = 'performance_history'
    
    id = Column(Integer, primary_key=True)
    video_id = Column(String(20), nullable=False, index=True)
    hours_after_publish = Column(Integer, nullable=False)
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    collected_at = Column(DateTime, default=datetime.utcnow)


class BanditState(Base):
    """Contextual bandit algorithm state."""
    
    __tablename__ = 'bandit_state'
    
    id = Column(Integer, primary_key=True)
    context_key = Column(String(100), nullable=False, index=True)
    hour_of_day = Column(Integer, nullable=False)
    alpha = Column(Float, default=1.0)
    beta = Column(Float, default=1.0)
    total_pulls = Column(Integer, default=0)
    total_reward = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow)


class Recommendation(Base):
    """Generated posting recommendations."""
    
    __tablename__ = 'recommendations'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, index=True)
    hour_of_day = Column(Integer, nullable=False)
    expected_views = Column(Float)
    confidence_score = Column(Float)
    context_features = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Database operations manager."""
    
    def __init__(self, database_url: str):
        """Initialize database connection.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
        
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
        
    def store_videos(self, videos_data: List[Dict]) -> int:
        """Store video data in database.
        
        Args:
            videos_data: List of video data dictionaries
            
        Returns:
            Number of videos stored
        """
        session = self.get_session()
        stored_count = 0
        
        try:
            for video_data in videos_data:
                # Check if video already exists
                existing = session.query(Video).filter_by(
                    video_id=video_data['video_id']
                ).first()
                
                if existing:
                    # Update existing video
                    for key, value in video_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                else:
                    # Create new video record
                    video = Video(**video_data)
                    session.add(video)
                    
                stored_count += 1
                
            session.commit()
            logger.info(f"Stored {stored_count} videos in database")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing videos: {e}")
            raise
        finally:
            session.close()
            
        return stored_count
    
    def get_videos_dataframe(self, channel_id: Optional[str] = None, 
                           days_back: Optional[int] = None) -> pd.DataFrame:
        """Get videos as pandas DataFrame.
        
        Args:
            channel_id: Filter by channel ID
            days_back: Only include videos from last N days
            
        Returns:
            DataFrame with video data
        """
        session = self.get_session()
        
        try:
            query = session.query(Video)
            
            if channel_id:
                query = query.filter(Video.channel_id == channel_id)
                
            if days_back:
                cutoff_date = datetime.utcnow() - pd.Timedelta(days=days_back)
                query = query.filter(Video.published_at >= cutoff_date)
                
            videos = query.all()
            
            # Convert to DataFrame
            data = []
            for video in videos:
                video_dict = {
                    'video_id': video.video_id,
                    'title': video.title,
                    'published_at': video.published_at,
                    'channel_id': video.channel_id,
                    'category_id': video.category_id,
                    'tags': video.tags,
                    'views': video.views,
                    'likes': video.likes,
                    'comments': video.comments,
                    'hour_of_day': video.hour_of_day,
                    'day_of_week': video.day_of_week,
                    'engagement_rate': video.engagement_rate,
                    'like_rate': video.like_rate
                }
                data.append(video_dict)
                
            return pd.DataFrame(data)
            
        finally:
            session.close()
    
    def store_performance_history(self, video_id: str, hours_after: int, 
                                metrics: Dict) -> None:
        """Store performance metrics at specific time after publish.
        
        Args:
            video_id: YouTube video ID
            hours_after: Hours after publish
            metrics: Performance metrics dictionary
        """
        session = self.get_session()
        
        try:
            performance = PerformanceHistory(
                video_id=video_id,
                hours_after_publish=hours_after,
                views=metrics.get('views', 0),
                likes=metrics.get('likes', 0),
                comments=metrics.get('comments', 0)
            )
            
            session.add(performance)
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing performance history: {e}")
            raise
        finally:
            session.close()
    
    def update_bandit_state(self, context_key: str, hour: int, 
                          reward: float) -> None:
        """Update contextual bandit state after observing reward.
        
        Args:
            context_key: Context identifier
            hour: Hour of day (0-23)
            reward: Observed reward
        """
        session = self.get_session()
        
        try:
            bandit_state = session.query(BanditState).filter_by(
                context_key=context_key,
                hour_of_day=hour
            ).first()
            
            if bandit_state:
                # Update existing state
                bandit_state.total_pulls += 1
                bandit_state.total_reward += reward
                bandit_state.last_updated = datetime.utcnow()
                
                # Update Beta distribution parameters
                if reward > 0:
                    bandit_state.alpha += 1
                else:
                    bandit_state.beta += 1
            else:
                # Create new state
                bandit_state = BanditState(
                    context_key=context_key,
                    hour_of_day=hour,
                    alpha=2.0 if reward > 0 else 1.0,
                    beta=1.0 if reward > 0 else 2.0,
                    total_pulls=1,
                    total_reward=reward
                )
                session.add(bandit_state)
                
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating bandit state: {e}")
            raise
        finally:
            session.close()
    
    def get_bandit_states(self, context_key: str) -> pd.DataFrame:
        """Get bandit states for a context.
        
        Args:
            context_key: Context identifier
            
        Returns:
            DataFrame with bandit states
        """
        session = self.get_session()
        
        try:
            states = session.query(BanditState).filter_by(
                context_key=context_key
            ).all()
            
            data = []
            for state in states:
                data.append({
                    'hour_of_day': state.hour_of_day,
                    'alpha': state.alpha,
                    'beta': state.beta,
                    'total_pulls': state.total_pulls,
                    'total_reward': state.total_reward,
                    'last_updated': state.last_updated
                })
                
            return pd.DataFrame(data)
            
        finally:
            session.close()