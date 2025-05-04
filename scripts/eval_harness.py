#!/usr/bin/env python3
"""
Evaluation harness for router accuracy using HTTP API and CSV eval set.
Requirements: requests, pandas, scikit-learn
"""
import argparse
import requests
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import json

# --- CLI Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate router accuracy via HTTP API.")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/api/v1/route", help="Router API endpoint URL")
    parser.add_argument("--csv", type=str, default="data/eval_set.csv", help="Path to evaluation CSV file")
    return parser.parse_args()

# --- Main Evaluation Logic ---
def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    y_true = []
    y_pred = []
    errors = []

    for idx, row in df.iterrows():
        query = row["query"]
        history = row["history"]
        label = row["label"]
        payload = {"query": query}
        # If history is a non-empty string, try to parse as JSON list, else skip
        if isinstance(history, str) and history.strip():
            try:
                payload["history"] = json.loads(history)
            except Exception:
                payload["history"] = history  # fallback: send as string
        try:
            resp = requests.post(args.api_url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            pred = data.get("route", "error")
        except Exception as e:
            pred = "error"
            errors.append({"idx": idx, "query": query, "error": str(e)})
        y_true.append(label)
        y_pred.append(pred)
        print(f"[{idx+1}/{len(df)}] True: {label:8} | Pred: {pred:8} | Query: {query[:60]}")

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, digits=3))

    print("\n--- Confusion Matrix ---")
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))

    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for err in errors:
            print(err)

if __name__ == "__main__":
    main() 