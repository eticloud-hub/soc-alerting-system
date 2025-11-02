import streamlit as st
import pandas as pd
import time
from ingestion_engine import IngestionEngine
from alert_manager import AlertManager
from anomaly_model import AnomalyDetector

# Page configuration
st.set_page_config(page_title="SOC Alerting System", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .anomaly-high { color: #ff4b4b; font-weight: bold; }
    .anomaly-medium { color: #ffa500; font-weight: bold; }
    .anomaly-low { color: #51cf66; font-weight: bold; }
    .header { font-size: 28px; font-weight: bold; color: #0f1419; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = IngestionEngine(batch_size=50)
    st.session_state.alert_manager = AlertManager()
    st.session_state.running = False

def get_severity_color(severity):
    """Return color based on severity level."""
    colors = {
        'CRITICAL': '🔴',
        'HIGH': '🟠',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }
    return colors.get(severity, '⚪')

# Main title
st.markdown('<div class="header">🔒 SOC Real-Time Alerting System</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
batch_size = st.sidebar.slider("Batch Size", min_value=10, max_value=500, value=100, step=10)
contamination = st.sidebar.slider("Anomaly Contamination (%)", min_value=1, max_value=10, value=5)
refresh_interval = st.sidebar.slider("Refresh Interval (sec)", min_value=1, max_value=10, value=2)

if st.sidebar.button("🔄 Retrain Model"):
    st.sidebar.info("Retraining anomaly detection model...")
    metrics = st.session_state.engine.retrain_model()
    if metrics:
        st.sidebar.success(f"✅ Model retrained on {metrics['samples_trained']} logs")
    else:
        st.sidebar.error("❌ Failed to retrain model")

if st.sidebar.button("📊 Export Alerts to CSV"):
    alerts = st.session_state.alert_manager.fetch_latest_alerts(limit=10000)
    if alerts:
        df = pd.DataFrame(alerts, columns=['ID', 'Timestamp', 'IP', 'User', 'Event', 'Score', 'Severity'])
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="alerts.csv",
            mime="text/csv"
        )
    else:
        st.sidebar.warning("No alerts to export")

st.sidebar.markdown("---")
st.sidebar.markdown("**System Info**")
model_info = st.session_state.engine.detector.get_model_info()

# Format threshold and status properly
threshold_str = f"{model_info['normal_score_threshold']:.4f}" if model_info['normal_score_threshold'] else 'N/A'
status_str = '✅ Trained' if model_info['is_trained'] else '❌ Not Trained'
contamination_pct = f"{model_info['contamination']*100:.1f}%"

st.sidebar.info(f"**Model Status:** {status_str}\n"
                f"**Contamination:** {contamination_pct}\n"
                f"**Threshold:** {threshold_str}")

# Main dashboard layout
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🚨 Alerts", "📈 Analytics"])

# TAB 1: Dashboard Overview
with tab1:
    # Refresh metrics
    placeholder_metrics = st.empty()
    placeholder_status = st.empty()
    
    while True:
        metrics = st.session_state.engine.get_metrics()
        
        with placeholder_metrics.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="📥 Total Logs Processed",
                    value=f"{metrics['total_logs_processed']:,}"
                )
            
            with col2:
                st.metric(
                    label="🚨 Anomalies Detected",
                    value=f"{metrics['total_anomalies_detected']:,}"
                )
            
            with col3:
                st.metric(
                    label="⚡ Throughput",
                    value=f"{metrics['throughput_logs_per_sec']:.0f} logs/sec"
                )
            
            with col4:
                st.metric(
                    label="⏱️ Uptime",
                    value=f"{metrics['elapsed_time_sec']:.0f} sec"
                )
        
        with placeholder_status.container():
            st.subheader("System Status")
            col_status1, col_status2, col_status3 = st.columns(3)
            
            with col_status1:
                st.info(f"**Buffer Size:** {metrics['buffer_size']} logs")
            
            with col_status2:
                st.info(f"**Batches Processed:** {metrics['batch_count']}")
            
            with col_status3:
                anomaly_rate = (metrics['total_anomalies_detected'] / max(metrics['total_logs_processed'], 1)) * 100
                st.warning(f"**Anomaly Rate:** {anomaly_rate:.2f}%")
        
        time.sleep(refresh_interval)

# TAB 2: Real-Time Alerts
with tab2:
    st.subheader("🚨 Latest Alerts")
    
    alert_limit = st.slider("Show Last N Alerts", min_value=10, max_value=1000, value=50)
    
    placeholder_alerts = st.empty()
    
    while True:
        alerts = st.session_state.alert_manager.fetch_latest_alerts(limit=alert_limit)
        
        with placeholder_alerts.container():
            if alerts:
                df_alerts = pd.DataFrame(alerts, columns=['ID', 'Timestamp', 'IP', 'User', 'Event', 'Score', 'Severity'])
                
                # Color code by severity
                def style_severity(row):
                    severity = row['Severity']
                    if severity == 'CRITICAL':
                        return ['background-color: #ffcccc'] * len(row)
                    elif severity == 'HIGH':
                        return ['background-color: #ffe6cc'] * len(row)
                    elif severity == 'MEDIUM':
                        return ['background-color: #ffffcc'] * len(row)
                    else:
                        return ['background-color: #ccffcc'] * len(row)
                
                styled_df = df_alerts.style.apply(style_severity, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                
                # Summary stats
                severity_counts = df_alerts['Severity'].value_counts()
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("CRITICAL", severity_counts.get('CRITICAL', 0))
                with col2:
                    st.metric("HIGH", severity_counts.get('HIGH', 0))
                with col3:
                    st.metric("MEDIUM", severity_counts.get('MEDIUM', 0))
                with col4:
                    st.metric("LOW", severity_counts.get('LOW', 0))
            else:
                st.info("ℹ️ No alerts detected yet. System is monitoring...")
        
        time.sleep(refresh_interval)

# TAB 3: Analytics & Model Info
with tab3:
    st.subheader("📈 Model Performance")
    
    model_info = st.session_state.engine.detector.get_model_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Metadata**")
        
        # Format threshold properly before passing to st.json
        model_display = {
            'Type': model_info['model_type'],
            'Trained': model_info['is_trained'],
            'Estimators': model_info['n_estimators'],
            'Contamination': f"{model_info['contamination']*100:.1f}%",
            'Features': len(model_info['feature_names']) if model_info['feature_names'] else 0,
            'Threshold': f"{model_info['normal_score_threshold']:.4f}" if model_info['normal_score_threshold'] else 'N/A'
        }
        st.json(model_display)
    
    with col2:
        st.write("**Features Used**")
        if model_info['feature_names']:
            for i, feature in enumerate(model_info['feature_names'], 1):
                st.text(f"{i}. {feature}")
        else:
            st.warning("No features loaded")
    
    st.markdown("---")
    st.subheader("📊 Anomaly Distribution")
    
    alerts = st.session_state.alert_manager.fetch_latest_alerts(limit=500)
    
    if alerts:
        df_alerts = pd.DataFrame(alerts, columns=['ID', 'Timestamp', 'IP', 'User', 'Event', 'Score', 'Severity'])
        
        # Bar chart of severity distribution
        severity_counts = df_alerts['Severity'].value_counts()
        st.bar_chart(severity_counts)
        
        # Score distribution histogram
        st.subheader("Anomaly Score Distribution")
        scores = pd.to_numeric(df_alerts['Score'], errors='coerce')
        st.histogram(scores, bins=30, title="Score Distribution")
    else:
        st.info("No alerts to display analytics")

# Footer
st.markdown("---")
st.markdown(
    """
    **🔒 SOC Alerting System v1.0**
    
    Real-time security log anomaly detection using Isolation Forest ML model.
    Processes 1000+ logs/sec with sub-second latency.
    
    Built with: Python | Streamlit | Scikit-Learn | SQLite | Async I/O
    """
)
