#!/usr/bin/env python3
"""
Performance test script for router API.
Requirements: requests, numpy, psutil
"""
import argparse
import platform
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import psutil
import requests


# --- CLI Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Router API performance test.")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/api/v1/route", help="Router API endpoint URL")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests to send")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent workers (1=serial)")
    parser.add_argument("--query", type=str, default="What is the capital of France?", help="Query to use in requests")
    return parser.parse_args()

# --- System Info ---
def print_system_info():
    print("--- System Info ---")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"CPU: {platform.processor()} | Cores: {psutil.cpu_count(logical=True)}")
    print(f"RAM: {round(psutil.virtual_memory().total / 1e9, 2)} GB")
    print()

# --- Serial Test ---
def serial_test(api_url, n, query):
    print(f"Running serial test: {n} requests...")
    times = []
    errors = 0
    for i in range(n):
        payload = {"query": query}
        start = time.time()
        try:
            resp = requests.post(api_url, json=payload, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            errors += 1
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        print(f"[{i+1}/{n}] {elapsed:.1f} ms", end="\r")
    print()
    return times, errors

# --- Load Test ---
def load_test(api_url, n, concurrency, query):
    print(f"Running load test: {n} requests, {concurrency} workers...")
    times = []
    errors = 0
    def worker():
        payload = {"query": query}
        start = time.time()
        try:
            resp = requests.post(api_url, json=payload, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            nonlocal errors
            errors += 1
        elapsed = (time.time() - start) * 1000
        return elapsed
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(n)]
        for i, f in enumerate(as_completed(futures)):
            elapsed = f.result()
            times.append(elapsed)
            print(f"[{i+1}/{n}] {elapsed:.1f} ms", end="\r")
    print()
    return times, errors

# --- Main ---
def main():
    args = parse_args()
    print_system_info()
    if args.concurrency == 1:
        times, errors = serial_test(args.api_url, args.requests, args.query)
    else:
        times, errors = load_test(args.api_url, args.requests, args.concurrency, args.query)
    times = np.array(times)
    print("\n--- Latency Report (ms) ---")
    print(f"Mean:   {np.mean(times):.2f}")
    print(f"Median: {np.median(times):.2f}")
    print(f"p95:    {np.percentile(times, 95):.2f}")
    print(f"Min:    {np.min(times):.2f}")
    print(f"Max:    {np.max(times):.2f}")
    print(f"Errors: {errors} / {len(times)} ({100*errors/len(times):.1f}%)")
    if args.concurrency > 1:
        duration = np.sum(times) / 1000  # total time in seconds (approx)
        rps = len(times) / duration if duration > 0 else 0
        print(f"Approx RPS: {rps:.2f}")

if __name__ == "__main__":
    main() 