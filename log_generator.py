import asyncio
import json
import random
import time
from datetime import datetime

EVENT_TYPES = ['login', 'file_access', 'api_call', 'logout']
STATUS_CODES = [200, 403, 500]

def generate_log_entry():
    timestamp = datetime.utcnow().isoformat()
    ip = f"192.168.{random.randint(0,255)}.{random.randint(0,255)}"
    user_id = random.randint(1, 50)
    event_type = random.choices(EVENT_TYPES, weights=[0.4, 0.3, 0.2, 0.1])[0]
    response_time = random.gauss(100, 20)
    status_code = random.choices(STATUS_CODES, weights=[0.85, 0.1, 0.05])[0]

    # Inject anomalies
    if random.random() < 0.01:  # 1% anomaly chance
        event_type = 'failed_login'
        status_code = 403
        response_time *= 3

    return {
        "timestamp": timestamp,
        "ip": ip,
        "user_id": user_id,
        "event_type": event_type,
        "response_time": max(0, response_time),
        "status_code": status_code
    }

async def stream_logs(queue, rate_per_sec=1000):
    interval = 1.0 / rate_per_sec
    while True:
        log = generate_log_entry()
        await queue.put(log)
        await asyncio.sleep(interval)

# For testing independently:
if __name__ == "__main__":
    async def test_stream():
        queue = asyncio.Queue()
        asyncio.create_task(stream_logs(queue))
        while True:
            log = await queue.get()
            print(log)
            queue.task_done()

    asyncio.run(test_stream())
