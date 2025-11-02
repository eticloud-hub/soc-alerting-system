import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anomaly_model import AnomalyDetector
from feature_engineering import extract_features
import numpy as np

def test_model_initialization():
    """Test that anomaly detector initializes correctly."""
    detector = AnomalyDetector(model_path='test_init_model_xyz.pkl')
    assert not detector.is_trained, "Model should not be trained on init"
    assert detector.contamination == 0.05, "Default contamination should be 0.05"
    if os.path.exists('test_init_model_xyz.pkl'):
        os.remove('test_init_model_xyz.pkl')

def test_model_training():
    """Test model training on normal logs."""
    logs = [
        {
            'timestamp': f'2025-11-02T10:00:{i:02d}',
            'ip': '192.168.1.1',
            'user_id': 1,
            'event_type': 'login',
            'response_time': 100 + np.random.normal(0, 10),
            'status_code': 200
        }
        for i in range(50)
    ]
    
    features = extract_features(logs)
    detector = AnomalyDetector(model_path='test_train_model_xyz.pkl')
    metrics = detector.train(features)
    
    assert detector.is_trained, "Model should be trained after train()"
    assert metrics['samples_trained'] == 50
    assert metrics['features_used'] == 8
    assert detector.normal_score_threshold is not None
    if os.path.exists('test_train_model_xyz.pkl'):
        os.remove('test_train_model_xyz.pkl')

def test_model_prediction():
    """Test anomaly detection on mixed data."""
    normal_logs = [
        {
            'timestamp': f'2025-11-02T10:00:{i:02d}',
            'ip': '192.168.1.1',
            'user_id': 1,
            'event_type': 'login',
            'response_time': 100,
            'status_code': 200
        }
        for i in range(50)
    ]
    
    anomaly_log = {
        'timestamp': '2025-11-02T10:01:00',
        'ip': '10.0.0.1',
        'user_id': 99,
        'event_type': 'failed_login',
        'response_time': 500,
        'status_code': 403
    }
    
    normal_features = extract_features(normal_logs)
    detector = AnomalyDetector(model_path='test_pred_model_xyz.pkl')
    detector.train(normal_features)
    
    anomaly_features = extract_features([anomaly_log])
    scores, is_anomaly = detector.predict(anomaly_features)
    
    assert isinstance(scores, np.ndarray)
    assert isinstance(is_anomaly, np.ndarray)
    assert len(scores) > 0
    if os.path.exists('test_pred_model_xyz.pkl'):
        os.remove('test_pred_model_xyz.pkl')

def test_single_prediction():
    """Test real-time single event prediction."""
    normal_logs = [
        {
            'timestamp': f'2025-11-02T10:00:{i:02d}',
            'ip': '192.168.1.1',
            'user_id': 1,
            'event_type': 'login',
            'response_time': 100,
            'status_code': 200
        }
        for i in range(50)
    ]
    
    normal_features = extract_features(normal_logs)
    detector = AnomalyDetector(model_path='test_single_model_xyz.pkl')
    detector.train(normal_features)
    
    single_log = extract_features([{
        'timestamp': '2025-11-02T10:01:00',
        'ip': '192.168.1.1',
        'user_id': 1,
        'event_type': 'login',
        'response_time': 100,
        'status_code': 200
    }])
    
    score, is_anomaly = detector.predict_single(single_log)
    
    assert isinstance(score, (float, np.floating))
    assert isinstance(is_anomaly, (bool, np.bool_))
    if os.path.exists('test_single_model_xyz.pkl'):
        os.remove('test_single_model_xyz.pkl')

def test_severity_classification():
    """Test anomaly severity level classification."""
    normal_logs = [
        {
            'timestamp': f'2025-11-02T10:00:{i:02d}',
            'ip': '192.168.1.1',
            'user_id': 1,
            'event_type': 'login',
            'response_time': 100,
            'status_code': 200
        }
        for i in range(50)
    ]
    
    normal_features = extract_features(normal_logs)
    detector = AnomalyDetector(model_path='test_severity_model_xyz.pkl')
    detector.train(normal_features)
    
    threshold = detector.normal_score_threshold
    
    assert detector.get_anomaly_severity(threshold - 0.25) == 'CRITICAL'
    assert detector.get_anomaly_severity(threshold - 0.15) == 'HIGH'
    assert detector.get_anomaly_severity(threshold - 0.05) == 'MEDIUM'
    assert detector.get_anomaly_severity(threshold + 1) == 'LOW'
    if os.path.exists('test_severity_model_xyz.pkl'):
        os.remove('test_severity_model_xyz.pkl')

def test_model_persistence():
    """Test saving and loading trained model."""
    normal_logs = [
        {
            'timestamp': f'2025-11-02T10:00:{i:02d}',
            'ip': '192.168.1.1',
            'user_id': 1,
            'event_type': 'login',
            'response_time': 100,
            'status_code': 200
        }
        for i in range(50)
    ]
    
    normal_features = extract_features(normal_logs)
    detector1 = AnomalyDetector(model_path='test_persist_model_xyz.pkl')
    detector1.train(normal_features)
    
    detector2 = AnomalyDetector(model_path='test_persist_model_xyz.pkl')
    assert detector2.is_trained
    assert detector2.normal_score_threshold == detector1.normal_score_threshold
    if os.path.exists('test_persist_model_xyz.pkl'):
        os.remove('test_persist_model_xyz.pkl')
