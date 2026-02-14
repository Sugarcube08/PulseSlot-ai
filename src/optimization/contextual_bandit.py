"""Contextual bandit algorithms for optimal posting time selection."""

import logging
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import json

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ContextualBandit(ABC):
    """Abstract base class for contextual bandit algorithms."""
    
    @abstractmethod
    def select_arm(self, context: Dict, available_arms: List[int]) -> int:
        """Select an arm given context and available arms.
        
        Args:
            context: Context features dictionary
            available_arms: List of available arm indices
            
        Returns:
            Selected arm index
        """
        pass
    
    @abstractmethod
    def update(self, context: Dict, arm: int, reward: float) -> None:
        """Update bandit state with observed reward.
        
        Args:
            context: Context features dictionary
            arm: Selected arm index
            reward: Observed reward
        """
        pass
    
    @abstractmethod
    def get_arm_scores(self, context: Dict, available_arms: List[int]) -> Dict[int, float]:
        """Get confidence scores for all available arms.
        
        Args:
            context: Context features dictionary
            available_arms: List of available arm indices
            
        Returns:
            Dictionary mapping arm indices to confidence scores
        """
        pass


class ThompsonSamplingBandit(ContextualBandit):
    """Thompson Sampling contextual bandit with Beta-Bernoulli conjugate prior."""
    
    def __init__(self, n_arms: int = 24, alpha_prior: float = 1.0, 
                 beta_prior: float = 1.0):
        """Initialize Thompson Sampling bandit.
        
        Args:
            n_arms: Number of arms (hours of day: 0-23)
            alpha_prior: Prior alpha parameter for Beta distribution
            beta_prior: Prior beta parameter for Beta distribution
        """
        self.n_arms = n_arms
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        # Context-specific arm parameters
        self.context_arms = {}  # context_key -> {arm_id: {'alpha': float, 'beta': float}}
        
    def _get_context_key(self, context: Dict) -> str:
        """Generate context key from context features.
        
        Args:
            context: Context features dictionary
            
        Returns:
            String key representing context
        """
        # Create context key from relevant features
        key_features = [
            context.get('day_of_week', 0),
            context.get('is_weekend', 0),
            context.get('channel_id', 'unknown'),
            # Add more context features as needed
        ]
        return json.dumps(key_features, sort_keys=True)
    
    def _get_arm_params(self, context_key: str, arm: int) -> Tuple[float, float]:
        """Get Beta distribution parameters for a context-arm pair.
        
        Args:
            context_key: Context identifier
            arm: Arm index
            
        Returns:
            Tuple of (alpha, beta) parameters
        """
        if context_key not in self.context_arms:
            self.context_arms[context_key] = {}
            
        if arm not in self.context_arms[context_key]:
            self.context_arms[context_key][arm] = {
                'alpha': self.alpha_prior,
                'beta': self.beta_prior
            }
            
        params = self.context_arms[context_key][arm]
        return params['alpha'], params['beta']
    
    def select_arm(self, context: Dict, available_arms: List[int]) -> int:
        """Select arm using Thompson Sampling.
        
        Args:
            context: Context features dictionary
            available_arms: List of available arm indices
            
        Returns:
            Selected arm index
        """
        context_key = self._get_context_key(context)
        
        # Sample from posterior for each available arm
        arm_samples = {}
        for arm in available_arms:
            alpha, beta = self._get_arm_params(context_key, arm)
            # Sample from Beta distribution
            sample = np.random.beta(alpha, beta)
            arm_samples[arm] = sample
        
        # Select arm with highest sample
        selected_arm = max(arm_samples.keys(), key=lambda x: arm_samples[x])
        
        logger.debug(f"Thompson Sampling selected arm {selected_arm} "
                    f"with sample {arm_samples[selected_arm]:.4f}")
        
        return selected_arm
    
    def update(self, context: Dict, arm: int, reward: float) -> None:
        """Update Beta distribution parameters based on observed reward.
        
        Args:
            context: Context features dictionary
            arm: Selected arm index
            reward: Observed reward (should be normalized to [0,1] or binary)
        """
        context_key = self._get_context_key(context)
        
        # Get current parameters
        alpha, beta = self._get_arm_params(context_key, arm)
        
        # Update parameters based on reward
        # For continuous rewards, we can threshold or normalize
        success = 1 if reward > 0 else 0  # Simple binary conversion
        
        # Update Beta parameters
        self.context_arms[context_key][arm]['alpha'] = alpha + success
        self.context_arms[context_key][arm]['beta'] = beta + (1 - success)
        
        logger.debug(f"Updated arm {arm} parameters: "
                    f"alpha={self.context_arms[context_key][arm]['alpha']}, "
                    f"beta={self.context_arms[context_key][arm]['beta']}")
    
    def get_arm_scores(self, context: Dict, available_arms: List[int]) -> Dict[int, float]:
        """Get confidence scores (posterior means) for all available arms.
        
        Args:
            context: Context features dictionary
            available_arms: List of available arm indices
            
        Returns:
            Dictionary mapping arm indices to confidence scores
        """
        context_key = self._get_context_key(context)
        
        arm_scores = {}
        for arm in available_arms:
            alpha, beta = self._get_arm_params(context_key, arm)
            # Posterior mean of Beta distribution
            posterior_mean = alpha / (alpha + beta)
            arm_scores[arm] = posterior_mean
            
        return arm_scores


class LinUCBBandit(ContextualBandit):
    """Linear Upper Confidence Bound contextual bandit."""
    
    def __init__(self, n_arms: int = 24, n_features: int = 10, alpha: float = 1.0):
        """Initialize LinUCB bandit.
        
        Args:
            n_arms: Number of arms (hours of day: 0-23)
            n_features: Number of context features
            alpha: Exploration parameter
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        
        # Initialize parameters for each arm
        self.A = {}  # arm -> covariance matrix
        self.b = {}  # arm -> reward vector
        
        for arm in range(n_arms):
            self.A[arm] = np.identity(n_features)
            self.b[arm] = np.zeros(n_features)
    
    def _context_to_vector(self, context: Dict) -> np.ndarray:
        """Convert context dictionary to feature vector.
        
        Args:
            context: Context features dictionary
            
        Returns:
            Feature vector
        """
        # Extract relevant features and convert to vector
        features = [
            context.get('hour_of_day', 0) / 23.0,  # Normalize to [0,1]
            context.get('day_of_week', 0) / 6.0,
            context.get('is_weekend', 0),
            context.get('recent_avg_views', 0) / 1000000.0,  # Normalize
            context.get('recent_avg_engagement', 0),
            context.get('views_trend', 0),
            context.get('dow_avg_performance', 0) / 1000000.0,
            context.get('hour_avg_performance', 0) / 1000000.0,
            1.0  # Bias term
        ]
        
        # Pad or truncate to match n_features
        if len(features) < self.n_features:
            features.extend([0.0] * (self.n_features - len(features)))
        else:
            features = features[:self.n_features]
            
        return np.array(features)
    
    def select_arm(self, context: Dict, available_arms: List[int]) -> int:
        """Select arm using LinUCB algorithm.
        
        Args:
            context: Context features dictionary
            available_arms: List of available arm indices
            
        Returns:
            Selected arm index
        """
        x = self._context_to_vector(context)
        
        arm_ucbs = {}
        for arm in available_arms:
            # Calculate UCB for this arm
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            
            # Upper confidence bound
            confidence_width = self.alpha * np.sqrt(x.T @ A_inv @ x)
            ucb = x.T @ theta + confidence_width
            
            arm_ucbs[arm] = ucb
        
        # Select arm with highest UCB
        selected_arm = max(arm_ucbs.keys(), key=lambda x: arm_ucbs[x])
        
        logger.debug(f"LinUCB selected arm {selected_arm} "
                    f"with UCB {arm_ucbs[selected_arm]:.4f}")
        
        return selected_arm
    
    def update(self, context: Dict, arm: int, reward: float) -> None:
        """Update LinUCB parameters based on observed reward.
        
        Args:
            context: Context features dictionary
            arm: Selected arm index
            reward: Observed reward
        """
        x = self._context_to_vector(context)
        
        # Update covariance matrix and reward vector
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        
        logger.debug(f"Updated LinUCB parameters for arm {arm}")
    
    def get_arm_scores(self, context: Dict, available_arms: List[int]) -> Dict[int, float]:
        """Get confidence scores (expected rewards) for all available arms.
        
        Args:
            context: Context features dictionary
            available_arms: List of available arm indices
            
        Returns:
            Dictionary mapping arm indices to confidence scores
        """
        x = self._context_to_vector(context)
        
        arm_scores = {}
        for arm in available_arms:
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            expected_reward = x.T @ theta
            arm_scores[arm] = expected_reward
            
        return arm_scores


class PostingTimeOptimizer:
    """High-level optimizer for posting time selection using contextual bandits."""
    
    def __init__(self, bandit_algorithm: str = "thompson_sampling", 
                 bandit_params: Optional[Dict] = None):
        """Initialize posting time optimizer.
        
        Args:
            bandit_algorithm: Bandit algorithm to use
            bandit_params: Parameters for bandit algorithm
        """
        self.bandit_algorithm = bandit_algorithm
        self.bandit_params = bandit_params or {}
        
        # Initialize bandit
        if bandit_algorithm == "thompson_sampling":
            self.bandit = ThompsonSamplingBandit(**self.bandit_params)
        elif bandit_algorithm == "linucb":
            self.bandit = LinUCBBandit(**self.bandit_params)
        else:
            raise ValueError(f"Unknown bandit algorithm: {bandit_algorithm}")
    
    def recommend_posting_times(self, context: Dict, top_k: int = 3,
                              exclude_hours: Optional[List[int]] = None) -> List[Dict]:
        """Recommend top posting times with confidence scores.
        
        Args:
            context: Context features dictionary
            top_k: Number of recommendations to return
            exclude_hours: Hours to exclude from recommendations
            
        Returns:
            List of recommendation dictionaries
        """
        # Available hours (0-23)
        available_hours = list(range(24))
        if exclude_hours:
            available_hours = [h for h in available_hours if h not in exclude_hours]
        
        # Get confidence scores for all available hours
        arm_scores = self.bandit.get_arm_scores(context, available_hours)
        
        # Sort by confidence score
        sorted_arms = sorted(arm_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create recommendations
        recommendations = []
        for i, (hour, score) in enumerate(sorted_arms[:top_k]):
            recommendations.append({
                'hour_of_day': hour,
                'confidence_score': float(score),
                'rank': i + 1,
                'time_display': f"{hour:02d}:00"
            })
        
        return recommendations
    
    def update_performance(self, context: Dict, hour: int, 
                         actual_views: int, baseline_views: Optional[int] = None) -> None:
        """Update bandit with observed performance.
        
        Args:
            context: Context features used for recommendation
            hour: Hour that was selected
            actual_views: Actual views received
            baseline_views: Baseline views for normalization
        """
        # Calculate reward (normalized performance)
        if baseline_views and baseline_views > 0:
            reward = (actual_views - baseline_views) / baseline_views
        else:
            # Use log-normalized views as reward
            reward = np.log1p(actual_views) / 20.0  # Scale to reasonable range
        
        # Update bandit
        self.bandit.update(context, hour, reward)
        
        logger.info(f"Updated bandit with reward {reward:.4f} for hour {hour}")
    
    def get_weekly_heatmap_data(self, context_base: Dict) -> pd.DataFrame:
        """Generate weekly heatmap data showing confidence scores.
        
        Args:
            context_base: Base context features
            
        Returns:
            DataFrame with day_of_week, hour_of_day, confidence_score
        """
        heatmap_data = []
        
        for day_of_week in range(7):  # Monday=0 to Sunday=6
            context = context_base.copy()
            context['day_of_week'] = day_of_week
            context['is_weekend'] = 1 if day_of_week >= 5 else 0
            
            # Get scores for all hours
            available_hours = list(range(24))
            arm_scores = self.bandit.get_arm_scores(context, available_hours)
            
            for hour, score in arm_scores.items():
                heatmap_data.append({
                    'day_of_week': day_of_week,
                    'hour_of_day': hour,
                    'confidence_score': score
                })
        
        return pd.DataFrame(heatmap_data)
    
    def save_state(self, filepath: str) -> None:
        """Save bandit state to file.
        
        Args:
            filepath: Path to save state
        """
        state_data = {
            'bandit_algorithm': self.bandit_algorithm,
            'bandit_params': self.bandit_params,
            'bandit_state': None
        }
        
        # Save algorithm-specific state
        if isinstance(self.bandit, ThompsonSamplingBandit):
            state_data['bandit_state'] = {
                'context_arms': self.bandit.context_arms,
                'alpha_prior': self.bandit.alpha_prior,
                'beta_prior': self.bandit.beta_prior
            }
        elif isinstance(self.bandit, LinUCBBandit):
            state_data['bandit_state'] = {
                'A': {k: v.tolist() for k, v in self.bandit.A.items()},
                'b': {k: v.tolist() for k, v in self.bandit.b.items()},
                'alpha': self.bandit.alpha
            }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
            
        logger.info(f"Bandit state saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load bandit state from file.
        
        Args:
            filepath: Path to load state from
        """
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        self.bandit_algorithm = state_data['bandit_algorithm']
        self.bandit_params = state_data['bandit_params']
        
        # Restore algorithm-specific state
        if self.bandit_algorithm == "thompson_sampling":
            self.bandit = ThompsonSamplingBandit(**self.bandit_params)
            if state_data['bandit_state']:
                self.bandit.context_arms = state_data['bandit_state']['context_arms']
                self.bandit.alpha_prior = state_data['bandit_state']['alpha_prior']
                self.bandit.beta_prior = state_data['bandit_state']['beta_prior']
                
        elif self.bandit_algorithm == "linucb":
            self.bandit = LinUCBBandit(**self.bandit_params)
            if state_data['bandit_state']:
                self.bandit.A = {int(k): np.array(v) for k, v in state_data['bandit_state']['A'].items()}
                self.bandit.b = {int(k): np.array(v) for k, v in state_data['bandit_state']['b'].items()}
                self.bandit.alpha = state_data['bandit_state']['alpha']
        
        logger.info(f"Bandit state loaded from {filepath}")