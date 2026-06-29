# gcp_connector.py
import os
import pandas as pd
import numpy as np

# We provide a clean mechanism to use mock or real GCP Client libraries.
# If credentials exist or user wants real GCP connection, they can configure it.
GCP_AVAILABLE = False
try:
    from google.cloud import bigquery
    from google.cloud import storage
    # Check if credentials exist
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GCP_PROJECT"):
        GCP_AVAILABLE = True
except ImportError:
    pass

class GCPConnector:
    def __init__(self, project_id=None, bucket_name=None):
        self.project_id = project_id or os.environ.get("GCP_PROJECT", "afdri-logistics-prod")
        self.bucket_name = bucket_name or os.environ.get("GCS_BUCKET", "afdri-route-logs")
        self.use_mock = not GCP_AVAILABLE
        
        if not self.use_mock:
            try:
                self.bq_client = bigquery.Client(project=self.project_id)
                self.storage_client = storage.Client(project=self.project_id)
            except Exception as e:
                print(f"Error initializing GCP clients, falling back to mock mode: {e}")
                self.use_mock = True

    def get_fleet_data(self):
        """
        Fetches vehicle fleet telemetry.
        Reads from BigQuery `fleet.driver_status` table or generates realistic mock data.
        """
        if not self.use_mock:
            try:
                query = f"SELECT vehicle_id, driver_name, lat, lon, capacity, status FROM `{self.project_id}.fleet.driver_status`"
                return self.bq_client.query(query).to_dataframe()
            except Exception as e:
                print(f"BigQuery error, falling back to mock data: {e}")
        
        # Mock Fleet Data (around Manhattan/Brooklyn area)
        np.random.seed(42)
        drivers = ["Marcus", "Sarah", "Devon", "Aisha", "Elena", "Carlos", "Yuki", "Chloe", "Kenji", "Liam"]
        fleet = []
        for i, name in enumerate(drivers):
            fleet.append({
                "vehicle_id": f"VEH-{100 + i}",
                "driver_name": name,
                "lat": np.random.uniform(40.68, 40.80),
                "lon": np.random.uniform(-74.02, -73.92),
                "capacity": np.random.choice([100, 120, 150, 200]),
                "status": np.random.choice(["Active", "Standby"], p=[0.8, 0.2])
            })
        return pd.DataFrame(fleet)

    def get_orders_data(self, n_orders=200):
        """
        Fetches package order logs.
        Reads from BigQuery `orders.active_deliveries` table or generates mock data.
        """
        if not self.use_mock:
            try:
                query = f"SELECT order_id, address, lat, lon, demand, time_window, priority FROM `{self.project_id}.orders.active_deliveries` LIMIT {n_orders}"
                return self.bq_client.query(query).to_dataframe()
            except Exception as e:
                print(f"BigQuery error, falling back to mock data: {e}")
                
        # Mock Orders Data
        np.random.seed(1337)
        orders = []
        for i in range(n_orders):
            orders.append({
                "order_id": f"ORD-{10000 + i}",
                "lat": np.random.uniform(40.65, 40.85),
                "lon": np.random.uniform(-74.05, -73.88),
                "demand": int(np.random.randint(5, 25)),
                "time_window": np.random.choice(["09:00 - 12:00", "12:00 - 15:00", "15:00 - 18:00", "18:00 - 21:00"]),
                "priority": np.random.choice(["High", "Medium", "Low"], p=[0.15, 0.55, 0.30])
            })
        return pd.DataFrame(orders)

    def export_routes_to_gcs(self, routes_df, filename="routes_dispatch_log.parquet"):
        """
        Saves optimized route schedules as Parquet and exports to Cloud Storage.
        """
        os.makedirs("gcs_bucket_simulation", exist_ok=True)
        local_path = os.path.join("gcs_bucket_simulation", filename)
        routes_df.to_parquet(local_path, index=False)
        
        if not self.use_mock:
            try:
                bucket = self.storage_client.bucket(self.bucket_name)
                blob = bucket.blob(f"dispatches/{filename}")
                blob.upload_from_filename(local_path)
                return f"gs://{self.bucket_name}/dispatches/{filename}"
            except Exception as e:
                return f"GCS Upload Failed: {e}. Saved locally to {local_path} (Simulation mode)"
        
        return f"gs://{self.bucket_name}/dispatches/{filename} (Simulated upload. Written to local path: {local_path})"
