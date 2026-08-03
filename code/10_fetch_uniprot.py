#!/usr/bin/env python3
"""Batch-fetch UniProt canonical sequences for all STRING genes in our graph.
Uses UniProt REST API with rate limiting (~0.3s per request → ~80 min for 16K genes).
Resume-safe: saves progress after every batch."""

import json, time, requests, pickle
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data/processed"
SEQ_FILE = DATA / "uniprot_sequences_full.json"
PROGRESS_FILE = DATA / "uniprot_fetch_progress.json"

def load_graph_genes():
    with open(DATA / "graph_data.pkl", "rb") as f:
        g = pickle.load(f)
    return list(g["gene_list"])

def fetch_sequence(gene_symbol, max_retries=3):
    """Fetch canonical isoform sequence for a human gene from UniProt."""
    for attempt in range(max_retries):
        try:
            url = (f"https://rest.uniprot.org/uniprotkb/search"
                   f"?query=gene:{gene_symbol}+AND+organism_id:9606+AND+reviewed:true"
                   f"&format=json&size=1")
            r = requests.get(url, timeout=15)
            if r.status_code == 429:
                time.sleep(5)
                continue
            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
                if results:
                    seq = results[0].get("sequence", {}).get("value", "")
                    if seq and len(seq) >= 10:
                        return seq[:1024]  # truncate to 1024 aa
            return None
        except Exception:
            time.sleep(2)
    return None

def main():
    genes = load_graph_genes()
    print(f"Total genes to fetch: {len(genes)}")
    
    # Load existing progress
    sequences = {}
    if SEQ_FILE.exists():
        with open(SEQ_FILE) as f:
            sequences = json.load(f)
        print(f"Resuming: {len(sequences)} already fetched")
    
    failed = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            failed = set(json.load(f).get("failed", []))
    
    batch_size = 50
    for i in range(0, len(genes), batch_size):
        batch = genes[i:i+batch_size]
        new_in_batch = 0
        
        for gene in batch:
            if gene in sequences or gene in failed:
                continue
            
            seq = fetch_sequence(gene)
            time.sleep(0.25)  # rate limit
            
            if seq:
                sequences[gene] = seq
                new_in_batch += 1
            else:
                failed.add(gene)
        
        # Save progress
        with open(SEQ_FILE, "w") as f:
            json.dump(sequences, f, indent=2)
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"failed": list(failed)}, f)
        
        pct = (i + len(batch)) / len(genes) * 100
        eta_min = (len(genes) - i - len(batch)) * 0.3 / 60
        print(f"  {min(i+len(batch), len(genes))}/{len(genes)} ({pct:.0f}%) "
              f"| fetched: {len(sequences)} | failed: {len(failed)} | ETA: {eta_min:.0f} min")
    
    print(f"\nDone: {len(sequences)}/{len(genes)} sequences ({len(sequences)/len(genes)*100:.1f}%)")
    print(f"Saved to {SEQ_FILE}")

if __name__ == "__main__":
    main()
