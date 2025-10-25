#!/usr/bin/env python3
import json
import os
import queue
import subprocess
import threading
import time
import argparse
from pathlib import Path
from collections import OrderedDict

# Global variables
Q = queue.Queue(maxsize=200)
SUCCESS = 0
FAILED = 0
SUCCESS_LOCK = threading.Lock()
STOP_EVENT = threading.Event()
TOKEN = os.getenv("GITHUB_TOKEN")

if not TOKEN:
    raise RuntimeError("Please set environment variable: export GITHUB_TOKEN=ghp_xxx")


def clone_one_repo(owner, repo_name, commit_sha, out_dir):
    """Clone a single repository to a specific commit"""
    global SUCCESS, FAILED
    
    dest = out_dir / f"{owner}_{repo_name}_{commit_sha}"
    
    # Skip if directory already exists
    if dest.exists():
        with SUCCESS_LOCK:
            SUCCESS += 1
            print(f"\r>>> Exists/Success {SUCCESS} repos", end="", flush=True)
        return True
    
    # Clone repository
    clone_url = f"https://{TOKEN}@github.com/{owner}/{repo_name}.git"
    cmd = ["git", "clone", clone_url, str(dest)]
    
    try:
        # Clone the entire repository first
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        
        # Checkout to specified commit
        checkout_cmd = ["git", "-C", str(dest), "checkout", commit_sha]
        subprocess.run(checkout_cmd, check=True, capture_output=True, timeout=60)
        
        with SUCCESS_LOCK:
            SUCCESS += 1
            print(f"\r>>> Success {SUCCESS} repos", end="", flush=True)
        return True
        
    except subprocess.TimeoutExpired:
        print(f"\n[ERROR] Clone timeout: {owner}/{repo_name}")
        if dest.exists():
            subprocess.run(["rm", "-rf", str(dest)], capture_output=True)
    except Exception as e:
        print(f"\n[ERROR] Clone failed {owner}/{repo_name}@{commit_sha}: {e}")
        if dest.exists():
            subprocess.run(["rm", "-rf", str(dest)], capture_output=True)
    
    with SUCCESS_LOCK:
        FAILED += 1
    return False


def worker(out_dir, worker_id):
    """Worker thread"""
    while not STOP_EVENT.is_set():
        try:
            repo_info = Q.get(timeout=1)
        except queue.Empty:
            continue
        
        owner, repo_name, commit_sha = repo_info
        clone_one_repo(owner, repo_name, commit_sha, out_dir)
        
        # Random delay to avoid rate limiting
        time.sleep(0.5)
        Q.task_done()


def load_and_deduplicate_repos(train_json_path):
    """Load and deduplicate repository information from JSON"""
    print(f"Reading {train_json_path}...")
    
    with open(train_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original data: {len(data)} entries")
    
    # Use OrderedDict for deduplication while preserving order
    repos = OrderedDict()
    for item in data:
        owner = item['organization']
        repo_name = item['repo_name']
        commit = item['base_commit']
        
        # Use tuple as key for deduplication
        key = (owner, repo_name, commit)
        repos[key] = None
    
    repo_list = list(repos.keys())
    print(f"After deduplication: {len(repo_list)} unique repos")
    
    return repo_list


def main():
    parser = argparse.ArgumentParser(description="Extract and clone repositories from JSON file")
    parser.add_argument("--train_json", required=True, help="Path to training JSON file")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel worker threads")
    args = parser.parse_args()
    
    # Prepare output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and deduplicate repositories
    repo_list = load_and_deduplicate_repos(args.train_json)
    
    if not repo_list:
        print("No repositories found to clone")
        return
    
    print(f"\nStarting to clone {len(repo_list)} repos to {out_dir}")
    print(f"Using {args.workers} parallel threads\n")
    
    # Start worker threads
    threads = []
    for i in range(args.workers):
        t = threading.Thread(target=worker, args=(out_dir, i), daemon=True)
        t.start()
        threads.append(t)
    
    # Add all repositories to queue
    for repo_info in repo_list:
        Q.put(repo_info)
    
    # Wait for queue to be processed
    Q.join()
    
    # Stop all threads
    STOP_EVENT.set()
    for t in threads:
        t.join(timeout=2)
    
    print(f"\n\n🎉 Complete!")
    print(f"   Success: {SUCCESS} repos")
    print(f"   Failed: {FAILED} repos")
    print(f"   Total: {len(repo_list)} repos")


if __name__ == "__main__":
    main()