"""YouTube Data API client for collecting video metadata and engagement metrics."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd

logger = logging.getLogger(__name__)


class YouTubeAPIClient:
    """Client for interacting with YouTube Data API v3."""
    
    def __init__(self, api_key: str):
        """Initialize YouTube API client.
        
        Args:
            api_key: YouTube Data API key
        """
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        
    def get_channel_videos(self, channel_id: str, max_results: int = 50) -> List[Dict]:
        """Get recent videos from a channel.
        
        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos to retrieve
            
        Returns:
            List of video metadata dictionaries
        """
        videos = []
        next_page_token = None
        
        try:
            while len(videos) < max_results:
                # Get channel uploads playlist
                channel_response = self.youtube.channels().list(
                    part='contentDetails',
                    id=channel_id
                ).execute()
                
                if not channel_response['items']:
                    logger.warning(f"Channel {channel_id} not found")
                    return videos
                    
                uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
                
                # Get videos from uploads playlist
                playlist_response = self.youtube.playlistItems().list(
                    part='snippet',
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token
                ).execute()
                
                video_ids = [item['snippet']['resourceId']['videoId'] 
                           for item in playlist_response['items']]
                
                # Get detailed video statistics
                video_details = self._get_video_details(video_ids)
                videos.extend(video_details)
                
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
                    
                # Rate limiting
                time.sleep(0.1)
                
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            
        return videos[:max_results]
    
    def _get_video_details(self, video_ids: List[str]) -> List[Dict]:
        """Get detailed information for a list of video IDs.
        
        Args:
            video_ids: List of YouTube video IDs
            
        Returns:
            List of detailed video information
        """
        if not video_ids:
            return []
            
        try:
            response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=','.join(video_ids)
            ).execute()
            
            videos = []
            for item in response['items']:
                video_data = self._parse_video_item(item)
                if video_data:
                    videos.append(video_data)
                    
            return videos
            
        except HttpError as e:
            logger.error(f"Error getting video details: {e}")
            return []
    
    def _parse_video_item(self, item: Dict) -> Optional[Dict]:
        """Parse a video item from YouTube API response.
        
        Args:
            item: Video item from YouTube API
            
        Returns:
            Parsed video data dictionary
        """
        try:
            snippet = item['snippet']
            statistics = item['statistics']
            
            # Parse publish time
            published_at = datetime.fromisoformat(
                snippet['publishedAt'].replace('Z', '+00:00')
            )
            
            video_data = {
                'video_id': item['id'],
                'title': snippet['title'],
                'description': snippet.get('description', ''),
                'published_at': published_at,
                'channel_id': snippet['channelId'],
                'channel_title': snippet['channelTitle'],
                'category_id': int(snippet.get('categoryId', 0)),
                'tags': snippet.get('tags', []),
                'duration': item['contentDetails']['duration'],
                'views': int(statistics.get('viewCount', 0)),
                'likes': int(statistics.get('likeCount', 0)),
                'comments': int(statistics.get('commentCount', 0)),
                'collected_at': datetime.utcnow()
            }
            
            return video_data
            
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error parsing video item: {e}")
            return None
    
    def get_video_performance_history(self, video_id: str, 
                                    hours_after_publish: List[int] = [1, 6, 24]) -> Dict:
        """Get performance metrics at specific intervals after publishing.
        
        Args:
            video_id: YouTube video ID
            hours_after_publish: Hours after publish to collect metrics
            
        Returns:
            Dictionary with performance at different time intervals
        """
        try:
            response = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            ).execute()
            
            if not response['items']:
                return {}
                
            item = response['items'][0]
            current_stats = {
                'views': int(item['statistics'].get('viewCount', 0)),
                'likes': int(item['statistics'].get('likeCount', 0)),
                'comments': int(item['statistics'].get('commentCount', 0)),
                'collected_at': datetime.utcnow()
            }
            
            return current_stats
            
        except HttpError as e:
            logger.error(f"Error getting video performance: {e}")
            return {}