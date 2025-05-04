#!/usr/bin/env python3
"""
Demo script for the router service.
Sends example queries from each category to the API and prints the results.
Requirements: requests
"""
import requests
import argparse

EXAMPLES = [
    ("simple", "What is the capital of France?"),
    ("semantic", "Summarize the main findings from the latest research papers on quantum computing."),
    ("agent", "Help me design a research plan to study the effects of climate change on marine ecosystems."),
]

def main():
    parser = argparse.ArgumentParser(description="Demo the router API with example queries.")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/api/v1/route", help="Router API endpoint URL")
    args = parser.parse_args()
    print("--- Router Demo ---\n")
    for label, query in EXAMPLES:
        print(f"Category: {label}")
        print(f"Query:    {query}")
        try:
            resp = requests.post(args.api_url, json={"query": query}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            print(f"Response: {data}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main() 