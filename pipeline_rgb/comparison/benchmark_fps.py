#!/usr/bin/env python3
import time
import numpy as np
import open3d as o3d

# Mock data sizes based on your typical point clouds
N_SHELL = 2000
N_WALL  = 10000

def benchmark():
    print("Benchmarking Safety Check Kernel (KD-Tree)...")
    
    # Generate random mock data
    shell_pts = np.random.rand(N_SHELL, 3)
    wall_pts  = np.random.rand(N_WALL, 3)
    
    # Warmup
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(wall_pts)
    tree = o3d.geometry.KDTreeFlann(pcd)
    
    latencies = []
    
    for _ in range(50):
        start = time.time()
        
        # 1. Build Tree (Simulating per-frame dynamic walls)
        pcd.points = o3d.utility.Vector3dVector(wall_pts)
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        # 2. Query
        for i in range(0, N_SHELL, 5): # Stride 5 optimization
             tree.search_knn_vector_3d(shell_pts[i], 1)
             
        end = time.time()
        latencies.append((end - start) * 1000) # ms

    avg_ms = np.mean(latencies)
    fps = 1000.0 / avg_ms
    
    print(f"\nRESULTS:")
    print(f"Average Latency: {avg_ms:.2f} ms")
    print(f"Theoretical Max FPS: {fps:.1f} Hz")
    print("Conclusion: " + ("Real-time capable (>10Hz)" if fps > 10 else "Needs optimization"))

if __name__ == "__main__":
    benchmark()