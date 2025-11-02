import sqlite3
import datetime

class AlertManager:
    def __init__(self, db_path='alerts.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        sql = """CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                ip TEXT,
                user_id INTEGER,
                event_type TEXT,
                response_time REAL,
                status_code INTEGER,
                score REAL,
                severity TEXT
            )"""
        self.conn.execute(sql)
        self.conn.commit()

    def store_alert(self, log, score):
        severity = self._severity_level(score)
        sql = """INSERT INTO alerts (timestamp, ip, user_id, event_type, response_time, status_code, score, severity) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        self.conn.execute(sql, (
            log['timestamp'], log['ip'], log['user_id'], log['event_type'], 
            log['response_time'], log['status_code'], score, severity))
        self.conn.commit()

    def _severity_level(self, score):
        if score < -0.3:
            return 'HIGH'
        elif score < -0.1:
            return 'MEDIUM'
        else:
            return 'LOW'

    def fetch_latest_alerts(self, limit=20):
        cursor = self.conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
        return cursor.fetchall()
