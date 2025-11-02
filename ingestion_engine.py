import asyncio
import time
from collections import deque
from feature_engineering import extract_features
from anomaly_model import AnomalyDetector
from alert_manager import AlertManager
from log_generator import generate_log_entry


class IngestionEngine:
    """
    Real-time security log ingestion and anomaly detection pipeline.
    Processes logs at 1000+ events/sec with async buffering and ML scoring.
    """
    
    def __init__(self, batch_size=100, buffer_max_size=10000, model_path='anomaly_model.pkl'):
        """
        Initialize the ingestion engine.
        
        Args:
            batch_size (int): Number of logs to buffer before feature extraction
            buffer_max_size (int): Maximum logs to hold in memory
            model_path (str): Path to load/save trained anomaly model
        """
        self.batch_size = batch_size
        self.buffer_max_size = buffer_max_size
        self.buffer = deque(maxlen=buffer_max_size)
        
        # Initialize ML and storage components
        self.detector = AnomalyDetector(model_path=model_path)
        self.alert_manager = AlertManager()
        
        # Metrics tracking
        self.total_logs_processed = 0
        self.total_anomalies_detected = 0
        self.processing_start_time = time.time()
        self.batch_count = 0
        
        # Train model on startup with baseline data
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize anomaly detection model with baseline normal logs.
        """
        print("[*] Initializing anomaly detection model with baseline data...")
        
        # Generate baseline normal logs for training
        baseline_logs = [generate_log_entry() for _ in range(500)]
        baseline_features = extract_features(baseline_logs)
        
        if not baseline_features.empty:
            metrics = self.detector.train(baseline_features)
            print(f"[+] Model trained on {metrics['samples_trained']} baseline logs")
            print(f"[+] Anomaly threshold: {metrics['normal_score_threshold']:.4f}")
        else:
            print("[-] Warning: Could not generate baseline data")
    
    async def ingest_logs(self, log_queue):
        """
        Continuously consume logs from async queue.
        
        Args:
            log_queue (asyncio.Queue): Queue of incoming log entries
        """
        print("[*] Starting log ingestion pipeline...")
        
        try:
            while True:
                # Get log from queue (non-blocking with timeout)
                try:
                    log = await asyncio.wait_for(log_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Process accumulated buffer if no new logs arrive
                    if len(self.buffer) > 0:
                        await self._process_batch()
                    continue
                
                # Add to buffer
                self.buffer.append(log)
                self.total_logs_processed += 1
                
                # Process batch when buffer reaches threshold
                if len(self.buffer) >= self.batch_size:
                    await self._process_batch()
                
                log_queue.task_done()
        
        except KeyboardInterrupt:
            print("\n[!] Ingestion interrupted by user")
        except Exception as e:
            print(f"[-] Error in ingestion pipeline: {e}")
    
    async def _process_batch(self):
        """
        Extract features and run anomaly detection on buffered logs.
        """
        if len(self.buffer) == 0:
            return
        
        # Convert buffer to list and extract features
        batch_logs = list(self.buffer)
        self.buffer.clear()
        
        try:
            # Extract numeric features from raw logs
            features = extract_features(batch_logs)
            
            if features.empty:
                return
            
            # Run anomaly detection on batch
            scores, is_anomaly_flags = self.detector.predict(features)
            
            # Process each prediction
            for i, (log, score, is_anomaly) in enumerate(zip(batch_logs, scores, is_anomaly_flags)):
                if is_anomaly:
                    # Get severity level
                    severity = self.detector.get_anomaly_severity(score)
                    
                    # Store alert in database
                    self.alert_manager.store_alert(log, score)
                    self.total_anomalies_detected += 1
                    
                    # Log anomaly for monitoring
                    print(f"[!] ANOMALY DETECTED: {severity} | IP: {log['ip']} | User: {log['user_id']} | Score: {score:.4f}")
            
            self.batch_count += 1
            
            # Print periodic stats
            if self.batch_count % 100 == 0:
                self._print_stats()
        
        except Exception as e:
            print(f"[-] Error processing batch: {e}")
    
    def _print_stats(self):
        """Print ingestion pipeline performance statistics."""
        elapsed_time = time.time() - self.processing_start_time
        throughput = self.total_logs_processed / elapsed_time if elapsed_time > 0 else 0
        
        print(f"\n[STATS] Processed: {self.total_logs_processed} logs | "
              f"Anomalies: {self.total_anomalies_detected} | "
              f"Throughput: {throughput:.0f} logs/sec | "
              f"Batches: {self.batch_count}")
    
    def get_metrics(self):
        """
        Return current pipeline metrics for dashboard.
        
        Returns:
            dict: Metrics including processing stats and model info
        """
        elapsed_time = time.time() - self.processing_start_time
        throughput = self.total_logs_processed / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'total_logs_processed': self.total_logs_processed,
            'total_anomalies_detected': self.total_anomalies_detected,
            'throughput_logs_per_sec': throughput,
            'elapsed_time_sec': elapsed_time,
            'batch_count': self.batch_count,
            'buffer_size': len(self.buffer),
            'model_info': self.detector.get_model_info()
        }
    
    def retrain_model(self):
        """
        Manually retrain anomaly detection model with fresh baseline data.
        """
        print("[*] Retraining anomaly detection model...")
        baseline_logs = [generate_log_entry() for _ in range(500)]
        baseline_features = extract_features(baseline_logs)
        
        if not baseline_features.empty:
            metrics = self.detector.train(baseline_features)
            print(f"[+] Model retrained on {metrics['samples_trained']} logs")
            return metrics
        return None


async def main():
    """
    Main entry point: start log generator and ingestion engine together.
    """
    from log_generator import stream_logs
    
    # Create async queue for log streaming
    log_queue = asyncio.Queue(maxsize=10000)
    
    # Initialize ingestion engine
    engine = IngestionEngine(batch_size=50)
    
    # Run log producer and consumer concurrently
    producer = asyncio.create_task(stream_logs(log_queue, rate_per_sec=1000))
    consumer = asyncio.create_task(engine.ingest_logs(log_queue))
    
    print("[+] SOC Alerting System started")
    print("[+] Streaming 1000 logs/sec with anomaly detection enabled\n")
    
    try:
        await asyncio.gather(producer, consumer)
    except KeyboardInterrupt:
        print("\n[!] Shutting down...")
        producer.cancel()
        consumer.cancel()


if __name__ == "__main__":
    asyncio.run(main())
