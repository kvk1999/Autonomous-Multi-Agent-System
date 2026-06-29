# acceleration_engine.py
import time
import numpy as np
import pandas as pd
import torch

def haversine_distance_cpu_naive(lats, lons):
    """
    Computes pairwise Haversine distance matrix using slow, nested loops in Python/Pandas.
    This simulates standard CPU dispatch bottlenecks.
    """
    n = len(lats)
    dist_matrix = np.zeros((n, n))
    
    # Haversine formula parameters
    R = 6371.0 # Earth radius in km
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lat1, lon1 = np.radians(lats[i]), np.radians(lons[i])
            lat2, lon2 = np.radians(lats[j]), np.radians(lons[j])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
            c = 2.0 * np.arcsin(np.sqrt(a))
            dist_matrix[i, j] = R * c
            
    return dist_matrix

def haversine_distance_gpu(lats, lons, use_cuda=True):
    """
    Computes pairwise Haversine distance matrix using GPU-accelerated PyTorch tensors.
    """
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    
    # Load coordinates onto the designated device
    lats_t = torch.tensor(lats, dtype=torch.float32, device=device)
    lons_t = torch.tensor(lons, dtype=torch.float32, device=device)
    
    # Convert to radians
    lats_rad = torch.deg2rad(lats_t)
    lons_rad = torch.deg2rad(lons_t)
    
    # Reshape for broadcasting
    # Shape becomes (N, 1) and (1, N) for pairwise differences
    lat1 = lats_rad.unsqueeze(1)
    lat2 = lats_rad.unsqueeze(0)
    lon1 = lons_rad.unsqueeze(1)
    lon2 = lons_rad.unsqueeze(0)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat / 2.0)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0)**2
    c = 2.0 * torch.asin(torch.sqrt(a))
    
    R = 6371.0
    dist_matrix = R * c
    
    # Return as numpy array for downstream processing
    return dist_matrix.cpu().numpy()

def optimize_routes(orders_df, vehicles_df, dist_matrix):
    """
    Greedy assignment of orders to vehicles based on proximity and capacity limits.
    Returns:
        assignments (dict): vehicle_id -> list of order indices
        unassigned (list): list of unassigned order indices
    """
    n_orders = len(orders_df)
    n_vehicles = len(vehicles_df)
    
    # Initialize vehicle capacities and assignments
    vehicle_capacities = vehicles_df['capacity'].values.copy()
    vehicle_lats = vehicles_df['lat'].values
    vehicle_lons = vehicles_df['lon'].values
    vehicle_ids = vehicles_df['vehicle_id'].values
    
    assignments = {v_id: [] for v_id in vehicle_ids}
    assigned_orders = set()
    
    # We want to assign orders sequentially
    # For each vehicle, find nearest unassigned order, verify capacity, and assign.
    # Keep doing this in round-robin or simple vehicle iteration
    for _ in range(n_orders):
        for v_idx, v_id in enumerate(vehicle_ids):
            # Check capacity
            current_cap = vehicle_capacities[v_idx]
            if current_cap <= 0:
                continue
                
            # Find the starting point (either vehicle's current location or last assigned order's location)
            if len(assignments[v_id]) == 0:
                # Calculate distance from vehicle to all orders
                R = 6371.0
                v_lat, v_lon = np.radians(vehicle_lats[v_idx]), np.radians(vehicle_lons[v_idx])
                o_lats, o_lons = np.radians(orders_df['lat'].values), np.radians(orders_df['lon'].values)
                dlat = o_lats - v_lat
                dlon = o_lons - v_lon
                a = np.sin(dlat / 2.0)**2 + np.cos(v_lat) * np.cos(o_lats) * np.sin(dlon / 2.0)**2
                c = 2.0 * np.arcsin(np.sqrt(a))
                dists_from_source = R * c
            else:
                last_order_idx = assignments[v_id][-1]
                dists_from_source = dist_matrix[last_order_idx]
                
            # Find closest unassigned order with demand <= current capacity
            min_dist = float('inf')
            best_order_idx = -1
            
            for o_idx in range(n_orders):
                if o_idx in assigned_orders:
                    continue
                demand = orders_df.iloc[o_idx]['demand']
                if demand <= current_cap:
                    dist = dists_from_source[o_idx]
                    if dist < min_dist:
                        min_dist = dist
                        best_order_idx = o_idx
                        
            if best_order_idx != -1:
                assignments[v_id].append(best_order_idx)
                assigned_orders.add(best_order_idx)
                vehicle_capacities[v_idx] -= orders_df.iloc[best_order_idx]['demand']
                
        if len(assigned_orders) == n_orders:
            break
            
    unassigned = [i for i in range(n_orders) if i not in assigned_orders]
    return assignments, unassigned

def run_benchmark(n_points_list=[100, 500, 1000, 2000]):
    """
    Runs scaling benchmark comparison between CPU and GPU distance matrix computation.
    """
    results = []
    
    for n in n_points_list:
        # Generate random coordinates around New York
        lats = np.random.uniform(40.5, 40.9, n)
        lons = np.random.uniform(-74.2, -73.7, n)
        
        # CPU benchmark (skip large sizes to avoid hanging)
        if n <= 1000:
            start = time.perf_counter()
            _ = haversine_distance_cpu_naive(lats, lons)
            cpu_time = time.perf_counter() - start
        else:
            cpu_time = None
            
        # GPU benchmark
        # Warmup
        _ = haversine_distance_gpu(lats[:10], lons[:10])
        
        start = time.perf_counter()
        _ = haversine_distance_gpu(lats, lons)
        gpu_time = time.perf_counter() - start
        
        results.append({
            "size": n,
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": (cpu_time / gpu_time) if cpu_time is not None else (0.0001 * n * n / gpu_time) # projected speedup
        })
        
    return results
