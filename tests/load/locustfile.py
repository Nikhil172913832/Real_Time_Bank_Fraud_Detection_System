"""Load testing with Locust."""

from locust import HttpUser, task, between
import random
import json


class FraudDetectionUser(HttpUser):
    """Simulate users making fraud detection requests."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a simulated user starts."""
        self.transaction_id = 0
    
    @task(10)
    def predict_transaction(self):
        """Make a single prediction request."""
        self.transaction_id += 1
        
        transaction = {
            "transaction_id": f"tx_{self.transaction_id}",
            "amount": random.uniform(10, 5000),
            "source": random.choice(["online", "atm", "pos"]),
            "device_os": random.choice(["iOS", "Android", "Windows"]),
            "merchant_category": random.choice(["retail", "food", "entertainment"]),
            "is_international": random.choice([True, False]),
            "hour_of_day": random.randint(0, 23)
        }
        
        with self.client.post(
            "/predict",
            json=transaction,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def batch_predict(self):
        """Make a batch prediction request."""
        batch_size = random.randint(2, 10)
        transactions = []
        
        for i in range(batch_size):
            self.transaction_id += 1
            transactions.append({
                "transaction_id": f"tx_{self.transaction_id}",
                "amount": random.uniform(10, 5000),
                "source": random.choice(["online", "atm", "pos"])
            })
        
        with self.client.post(
            "/predict/batch",
            json=transactions,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Check API health."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def get_model_info(self):
        """Get model information."""
        with self.client.get("/model/info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Model info failed: {response.status_code}")
