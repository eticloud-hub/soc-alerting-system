import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from alert_manager import AlertManager

def test_alert_crud():
    # Use in-memory SQLite database to avoid file creation
    mgr = AlertManager(':memory:')
    
    sample_log = {
        'timestamp': '2025-11-02T10:00:00',
        'ip': '192.168.1.1',
        'user_id': 1,
        'event_type': 'failed_login',
        'response_time': 300,
        'status_code': 403
    }
    
    # Store an alert
    mgr.store_alert(sample_log, -0.5)
    
    # Fetch the latest alerts and test assertions
    alerts = mgr.fetch_latest_alerts()
    assert len(alerts) == 1
    assert alerts[0][-1] == 'HIGH'  # Severity level column
