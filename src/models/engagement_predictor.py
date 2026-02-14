"""Engagement prediction model using XGBoost with uncertainty estimation."""

import logging
from typing import Dict, List, Tuple, Optional
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from scipy import stats

logger = logging.getLogger(__name__)


class EngagementPredictor:
    """XGBoost-based engagement prediction model with uncertainty estimation."""
    
    def __init__(self, model_params: Optional[Dict] = None):
        """Initialize engagement predictor.
        
        Args:
            model_params: XGBoost model parameters
        """
        self.model_params = model_params or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'reg:squarederror'
        }
        
        self.model = xgb.XGBRegressor(**self.model_params)
        self.uncertainty_models = []  # For bootstrap uncertainty estimation
        self.feature_names = None
        self.target_scaler_params = None
        self.is_fitted = False
        
    def _prepare_features(self, df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        """Prepare feature matrix for training/prediction.
        
        Args:
            df: Feature DataFrame
            embeddings: Title embeddings
            
        Returns:
            Combined feature matrix
        """
        # Select numerical features (excluding target and metadata)
        exclude_cols = ['video_id', 'title', 'description', 'published_at', 
                       'channel_id', 'channel_title', 'tags', 'views', 'collected_at']
        
        numerical_features = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_features if col not in exclude_cols]
        
        # Get numerical features
        X_numerical = df[feature_cols].values
        
        # Combine with embeddings
        X_combined = np.hstack([X_numerical, embeddings])
        
        # Store feature names for later use
        embedding_names = [f'embedding_{i}' for i in range(embeddings.shape[1])]
        self.feature_names = feature_cols + embedding_names
        
        return X_combined
    
    def _prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare target variable (log-transformed views).
        
        Args:
            df: DataFrame with views column
            
        Returns:
            Log-transformed target values
        """
        # Use log(views + 1) as target to handle skewness
        y = np.log1p(df['views'].values)
        
        # Store scaling parameters for inverse transform
        self.target_scaler_params = {
            'mean': np.mean(y),
            'std': np.std(y)
        }
        
        return y
    
    def fit(self, df: pd.DataFrame, embeddings: np.ndarray, 
            n_bootstrap: int = 10) -> Dict:
        """Fit engagement prediction model with uncertainty estimation.
        
        Args:
            df: Training data DataFrame
            embeddings: Title embeddings
            n_bootstrap: Number of bootstrap models for uncertainty
            
        Returns:
            Training metrics dictionary
        """
        logger.info("Training engagement prediction model")
        
        # Prepare features and target
        X = self._prepare_features(df, embeddings)
        y = self._prepare_target(df)
        
        # Time series split for validation (no random leakage)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train main model
        self.model.fit(X, y)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=tscv, 
                                  scoring='neg_mean_squared_error')
        
        # Train bootstrap models for uncertainty estimation
        logger.info(f"Training {n_bootstrap} bootstrap models for uncertainty")
        self.uncertainty_models = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            n_samples = len(X)
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_idx]
            y_bootstrap = y[bootstrap_idx]
            
            # Train bootstrap model
            bootstrap_model = xgb.XGBRegressor(**self.model_params)
            bootstrap_model.fit(X_bootstrap, y_bootstrap)
            self.uncertainty_models.append(bootstrap_model)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
            'cv_rmse_std': np.sqrt(cv_scores.std())
        }
        
        self.is_fitted = True
        logger.info(f"Model training completed. RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame, embeddings: np.ndarray, 
                return_uncertainty: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict engagement with uncertainty estimation.
        
        Args:
            df: Feature DataFrame
            embeddings: Title embeddings
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._prepare_features(df, embeddings)
        
        # Main prediction
        y_pred = self.model.predict(X)
        
        # Convert back from log space
        views_pred = np.expm1(y_pred)
        
        if not return_uncertainty or not self.uncertainty_models:
            return views_pred, None
        
        # Bootstrap predictions for uncertainty
        bootstrap_preds = []
        for model in self.uncertainty_models:
            bootstrap_pred = model.predict(X)
            bootstrap_preds.append(np.expm1(bootstrap_pred))
        
        bootstrap_preds = np.array(bootstrap_preds)
        
        # Calculate uncertainty as standard deviation of bootstrap predictions
        uncertainties = np.std(bootstrap_preds, axis=0)
        
        return views_pred, uncertainties
    
    def predict_with_confidence(self, df: pd.DataFrame, embeddings: np.ndarray,
                              confidence_level: float = 0.95) -> Dict:
        """Predict with confidence intervals.
        
        Args:
            df: Feature DataFrame
            embeddings: Title embeddings
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        predictions, uncertainties = self.predict(df, embeddings, return_uncertainty=True)
        
        if uncertainties is None:
            return {
                'predictions': predictions,
                'lower_bound': predictions,
                'upper_bound': predictions,
                'uncertainty': np.zeros_like(predictions)
            }
        
        # Calculate confidence intervals
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * uncertainties
        
        return {
            'predictions': predictions,
            'lower_bound': predictions - margin_of_error,
            'upper_bound': predictions + margin_of_error,
            'uncertainty': uncertainties
        }
    
    def get_feature_importance(self, top_k: int = 20) -> pd.DataFrame:
        """Get feature importance scores.
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance_scores = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_k)
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'uncertainty_models': self.uncertainty_models,
            'feature_names': self.feature_names,
            'target_scaler_params': self.target_scaler_params,
            'model_params': self.model_params
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from disk.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.uncertainty_models = model_data['uncertainty_models']
        self.feature_names = model_data['feature_names']
        self.target_scaler_params = model_data['target_scaler_params']
        self.model_params = model_data['model_params']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def evaluate_time_series(self, df: pd.DataFrame, embeddings: np.ndarray,
                           test_size: float = 0.2) -> Dict:
        """Evaluate model using time series split.
        
        Args:
            df: Full dataset DataFrame
            embeddings: Title embeddings
            test_size: Fraction of data to use for testing
            
        Returns:
            Evaluation metrics dictionary
        """
        X = self._prepare_features(df, embeddings)
        y = self._prepare_target(df)
        
        # Time-based split (most recent data as test)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train on training set
        temp_model = xgb.XGBRegressor(**self.model_params)
        temp_model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = temp_model.predict(X_test)
        
        # Convert back to original scale
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
        
        # Calculate metrics
        metrics = {
            'test_rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'test_mae': mean_absolute_error(y_test_orig, y_pred_orig),
            'test_r2': r2_score(y_test_orig, y_pred_orig),
            'test_mape': np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + 1))) * 100
        }
        
        return metrics