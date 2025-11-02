import pandas as pd
import numpy as np

def extract_features(logs):
    """
    Convert raw security logs into numeric feature vectors for ML anomaly detection.
    
    Args:
        logs (list): List of log dictionaries with keys: timestamp, ip, user_id, event_type, response_time, status_code
    
    Returns:
        pd.DataFrame: Numeric feature matrix ready for ML model input
    """
    if not logs:
        return pd.DataFrame()
    
    df = pd.DataFrame(logs)
    
    # Ensure required columns exist
    required_cols = ['user_id', 'response_time', 'status_code', 'ip', 'event_type']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Feature 1: Response time (raw value, normalized)
    df['response_time'] = pd.to_numeric(df['response_time'], errors='coerce').fillna(0)
    df['response_time_normalized'] = (df['response_time'] - df['response_time'].mean()) / (df['response_time'].std() + 1e-5)
    
    # Feature 2: Status code encoding
    df['status_code'] = pd.to_numeric(df['status_code'], errors='coerce').fillna(200)
    df['is_error_status'] = (df['status_code'] >= 400).astype(int)  # 1 if error, 0 if success
    
    # Feature 3: User event frequency in batch
    user_freq = df['user_id'].value_counts().to_dict()
    df['user_event_count'] = df['user_id'].map(user_freq).fillna(0)
    
    # Feature 4: Unusual IP detection (detect IPs with low frequency)
    ip_freq = df['ip'].value_counts().to_dict()
    df['ip_event_count'] = df['ip'].map(ip_freq).fillna(0)
    df['is_rare_ip'] = (df['ip_event_count'] <= 1).astype(int)  # Rare IPs are suspicious
    
    # Feature 5: Event type encoding (one-hot or ordinal)
    event_type_mapping = {
        'login': 1,
        'logout': 2,
        'file_access': 3,
        'api_call': 4,
        'failed_login': 5  # High risk
    }
    df['event_type_encoded'] = df['event_type'].map(event_type_mapping).fillna(0)
    
    # Feature 6: Failed login indicator
    df['is_failed_login'] = (df['event_type'] == 'failed_login').astype(int)
    
    # Feature 7: Bulk activity indicator (high frequency from same user in small window)
    df['user_burst_activity'] = (df['user_event_count'] > df['user_event_count'].quantile(0.9)).astype(int)
    
    # Select final numeric feature columns for ML
    feature_cols = [
        'response_time_normalized',
        'is_error_status',
        'user_event_count',
        'ip_event_count',
        'is_rare_ip',
        'event_type_encoded',
        'is_failed_login',
        'user_burst_activity'
    ]
    
    # Return only numeric features
    return df[feature_cols].fillna(0).astype(float)


def get_feature_names():
    """
    Return feature column names for interpretability.
    
    Returns:
        list: Names of features used in ML model
    """
    return [
        'response_time_normalized',
        'is_error_status',
        'user_event_count',
        'ip_event_count',
        'is_rare_ip',
        'event_type_encoded',
        'is_failed_login',
        'user_burst_activity'
    ]
