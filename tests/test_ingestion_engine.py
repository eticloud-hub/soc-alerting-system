import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion_engine import IngestionEngine
from log_generator import generate_log_entry
import asyncio

def test_ingestion_initialization():
    """Test that ingestion engine initializes correctly."""
    engine = IngestionEngine(model_path='test_ingest_init.pkl')
    assert engine.total_logs_processed == 0
    assert engine.total_anomalies_detected == 0
    assert engine.detector.is_trained
    if os.path.exists('test_ingest_init.pkl'):
        os.remove('test_ingest_init.pkl')

def test_ingestion_metrics():
    """Test that metrics are collected correctly."""
    engine = IngestionEngine(model_path='test_ingest_metrics.pkl')
    
    # Simulate processing some logs
    engine.total_logs_processed = 100
    engine.total_anomalies_detected = 5
    
    metrics = engine.get_metrics()
    
    assert metrics['total_logs_processed'] == 100
    assert metrics['total_anomalies_detected'] == 5
    assert 'throughput_logs_per_sec' in metrics
    assert 'model_info' in metrics
    if os.path.exists('test_ingest_metrics.pkl'):
        os.remove('test_ingest_metrics.pkl')

def test_batch_processing():
    """Test that batch processing works."""
    engine = IngestionEngine(batch_size=10, model_path='test_ingest_batch.pkl')
    
    # Generate test logs
    logs = [generate_log_entry() for _ in range(10)]
    
    # Add to buffer
    for log in logs:
        engine.buffer.append(log)
    
    assert len(engine.buffer) == 10
    if os.path.exists('test_ingest_batch.pkl'):
        os.remove('test_ingest_batch.pkl')

def test_model_retraining():
    """Test manual model retraining."""
    engine = IngestionEngine(model_path='test_ingest_retrain.pkl')
    
    # Retrain model
    metrics = engine.retrain_model()
    
    assert metrics is not None
    assert metrics['samples_trained'] == 500
    assert engine.detector.is_trained
    if os.path.exists('test_ingest_retrain.pkl'):
        os.remove('test_ingest_retrain.pkl')
