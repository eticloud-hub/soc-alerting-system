from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import pickle
import os


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detection model for security logs.
    Detects suspicious events with real-time scoring and persistence.
    """
    
    def __init__(self, contamination=0.05, model_path='anomaly_model.pkl'):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination (float): Expected proportion of anomalies in data (0.01-0.1)
            model_path (str): Path to save/load trained model
        """
        self.contamination = contamination
        self.model_path = model_path
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        self.normal_score_threshold = None
        
        # Try to load existing model
        self._load_model()
    
    def train(self, features_df):
        """
        Train the Isolation Forest model on normal/baseline logs.
        
        Args:
            features_df (pd.DataFrame): Numeric feature matrix (n_samples x n_features)
        
        Returns:
            dict: Training metrics including model performance stats
        """
        if features_df.empty:
            raise ValueError("Cannot train on empty features DataFrame")
        
        # Store feature names for later use
        self.feature_names = features_df.columns.tolist()
        
        # Scale features for better model performance
        scaled_features = self.scaler.fit_transform(features_df)
        
        # Train Isolation Forest
        self.model.fit(scaled_features)
        self.is_trained = True
        
        # Calculate threshold for anomaly scores
        scores = self.model.score_samples(scaled_features)
        self.normal_score_threshold = np.percentile(scores, 95)
        
        # Save trained model
        self._save_model()
        
        # Return training metrics
        metrics = {
            'samples_trained': features_df.shape[0],
            'features_used': features_df.shape[1],
            'contamination': self.contamination,
            'normal_score_threshold': float(self.normal_score_threshold),
            'feature_names': self.feature_names
        }
        
        return metrics
    
    def predict(self, features_df):
        """
        Predict anomalies on new log data in real-time.
        
        Args:
            features_df (pd.DataFrame): Numeric feature matrix for prediction
        
        Returns:
            tuple: (anomaly_scores, is_anomaly_flags)
                - anomaly_scores: Numpy array of anomaly scores
                - is_anomaly_flags: Boolean array (True if anomaly, False if normal)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if features_df.empty:
            return np.array([]), np.array([], dtype=bool)
        
        # Scale features using the training scaler
        scaled_features = self.scaler.transform(features_df)
        
        # Get anomaly scores
        scores = self.model.score_samples(scaled_features)
        
        # Flag as anomaly if score is below threshold
        is_anomaly = scores < self.normal_score_threshold
        
        return scores, is_anomaly
    
    def predict_single(self, features_df):
        """
        Predict anomaly for a single log event (for real-time ingestion).
        
        Args:
            features_df (pd.DataFrame): Single row feature matrix
        
        Returns:
            tuple: (anomaly_score, is_anomaly)
                - anomaly_score: Float anomaly score
                - is_anomaly: Boolean flag (True if anomalous)
        """
        if features_df.empty or features_df.shape[0] == 0:
            return 0.0, False
        
        scores, is_anomaly = self.predict(features_df)
        
        if len(scores) > 0:
            return float(scores[0]), bool(is_anomaly[0])
        return 0.0, False
    
    def get_anomaly_severity(self, score):
        """
        Convert anomaly score to severity level.
        
        Args:
            score (float): Anomaly score from model
        
        Returns:
            str: Severity level ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
        """
        if self.normal_score_threshold is None:
            return 'LOW'
            
        if score < self.normal_score_threshold - 0.2:
            return 'CRITICAL'
        elif score < self.normal_score_threshold - 0.1:
            return 'HIGH'
        elif score < self.normal_score_threshold:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _save_model(self):
        """Persist trained model and scaler to disk."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'normal_score_threshold': self.normal_score_threshold,
                'is_trained': self.is_trained
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            print(f"Warning: Could not save model to {self.model_path}: {e}")
    
    def _load_model(self):
        """Load previously trained model and scaler from disk."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.normal_score_threshold = model_data['normal_score_threshold']
                self.is_trained = model_data['is_trained']
            except Exception as e:
                print(f"Warning: Could not load model from {self.model_path}: {e}")
    
    def get_model_info(self):
        """Return model metadata and status."""
        return {
            'is_trained': self.is_trained,
            'model_type': 'IsolationForest',
            'contamination': self.contamination,
            'n_estimators': self.model.n_estimators,
            'feature_names': self.feature_names,
            'normal_score_threshold': float(self.normal_score_threshold) if self.normal_score_threshold else None,
            'model_path': self.model_path
        }
