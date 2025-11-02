import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_engineering import extract_features, get_feature_names
import pandas as pd

def test_extract_features_basic():
    """Test basic feature extraction with sample logs."""
    logs = [
        {
            'timestamp': '2025-11-02T10:00:00',
            'ip': '192.168.1.1',
            'user_id': 1,
            'event_type': 'login',
            'response_time': 100,
            'status_code': 200
        },
        {
            'timestamp': '2025-11-02T10:00:01',
            'ip': '192.168.1.1',
            'user_id': 1,
            'event_type': 'file_access',
            'response_time': 150,
            'status_code': 200
        },
        {
            'timestamp': '2025-11-02T10:00:02',
            'ip': '192.168.1.100',
            'user_id': 2,
            'event_type': 'failed_login',
            'response_time': 50,
            'status_code': 403
        }
    ]
    
    features = extract_features(logs)
    
    # Check output shape
    assert features.shape[0] == 3, f"Expected 3 rows, got {features.shape[0]}"
    assert features.shape[1] == 8, f"Expected 8 features, got {features.shape[1]}"
    
    # Check that all values are numeric
    assert (features.dtypes == 'float64').all(), "All features should be float64"
    
    # Check that failed_login is detected
    assert features.iloc[2]['is_failed_login'] == 1, "Failed login should be flagged"
    assert features.iloc[2]['is_error_status'] == 1, "Error status should be flagged"

def test_extract_features_empty():
    """Test that empty log list returns empty DataFrame."""
    features = extract_features([])
    assert features.empty, "Empty logs should return empty DataFrame"

def test_feature_names():
    """Test that feature names are returned correctly."""
    names = get_feature_names()
    assert len(names) == 8, f"Expected 8 feature names, got {len(names)}"
    assert isinstance(names, list), "Feature names should be a list"
    assert all(isinstance(name, str) for name in names), "All names should be strings"

def test_rare_ip_detection():
    """Test detection of rare/unusual IPs."""
    logs = [
        {'timestamp': '2025-11-02T10:00:00', 'ip': '192.168.1.1', 'user_id': 1, 'event_type': 'login', 'response_time': 100, 'status_code': 200},
        {'timestamp': '2025-11-02T10:00:01', 'ip': '192.168.1.1', 'user_id': 1, 'event_type': 'login', 'response_time': 100, 'status_code': 200},
        {'timestamp': '2025-11-02T10:00:02', 'ip': '10.0.0.1', 'user_id': 2, 'event_type': 'login', 'response_time': 100, 'status_code': 200},  # Rare IP
    ]
    
    features = extract_features(logs)
    
    # The rare IP (10.0.0.1) should be flagged
    assert features.iloc[2]['is_rare_ip'] == 1, "Rare IP should be flagged"
    assert features.iloc[0]['is_rare_ip'] == 0, "Common IP should not be flagged"
