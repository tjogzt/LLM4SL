"""
步骤1.5 PPI网络构建：STRING + BioGRID双数据库整合
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import json
import gzip
import requests
from pathlib import Path
import pandas as pd
import networkx as nx
from config import *


HUMAN_TAXON = "9606"

def download_string(taxon: str = HUMAN_TAXON, version: str = "12.0") -> Path:
    """下载STRING人类PPI数据（人类子集约200MB，全物种138GB）"""
    url = (f"https://stringdb-downloads.org/download/"
           f"protein.links.v{version}/{taxon}.protein.links.v{version}.txt.gz")
    output = DATA_EXTERNAL / f"string_human_v{version}_links.txt.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    
    if output.exists():
        print(f"[STRING] 文件已存在 ({output.stat().st_size/1e6:.0f} MB): {output}")
        return output
    
    print(f"[STRING] 下载人类PPI: {url} ...")
    r = requests.get(url, stream=True, timeout=300)
    total = int(r.headers.get('content-length', 0))
    downloaded = 0
    with open(output, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total and downloaded % (10*1024*1024) == 0:
                print(f"  进度: {downloaded/total*100:.0f}%")
    print(f"[STRING] 下载完成: {downloaded/1e6:.0f} MB")
    return output


def download_string_aliases(taxon: str = HUMAN_TAXON, version: str = "12.0") -> Path:
    """下载STRING人类蛋白质别名映射"""
    url = (f"https://stringdb-downloads.org/download/"
           f"protein.aliases.v{version}/{taxon}.protein.aliases.v{version}.txt.gz")
    output = DATA_EXTERNAL / f"string_human_v{version}_aliases.txt.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    
    if output.exists():
        return output
    
    print(f"[STRING] 下载人类别名映射...")
    r = requests.get(url, stream=True, timeout=300)
    with open(output, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    return output


def build_ppi_network(combined_score_threshold: float = STRING_COMBINED_SCORE_THRESHOLD
                      ) -> tuple[nx.Graph, dict]:
    """构建PPI网络：STRING主体 + BioGRID验证层"""
    print("[PPI] 构建蛋白质互作网络...")
    
    # === STRING主体 ===
    string_file = download_string()
    aliases_file = download_string_aliases()
    
    # 加载STRING别名映射
    print("[PPI] 加载STRING别名映射...")
    alias_map = {}  # string_id → gene_symbol
    with gzip.open(aliases_file, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                string_id = parts[0]
                source = parts[1]
                symbol = parts[2]
                if source == "Ensembl_HGNC_symbol" and not symbol.startswith("HGNC:"):
                    alias_map[string_id] = symbol
    
    print(f"[PPI] 加载别名: {len(alias_map):,} 条映射")
    
    # 构建网络
    G = nx.Graph()
    edges_added = 0
    
    with gzip.open(string_file, "rt") as f:
        for line in f:
            if line.startswith("protein1"):
                continue  # 跳过header
            parts = line.strip().split()
            p1, p2, score = parts[0], parts[1], float(parts[2])
            
            if score < combined_score_threshold * 1000:
                continue  # STRING分数是0-1000
            
            # 映射为gene symbol
            s1 = alias_map.get(p1, p1)
            s2 = alias_map.get(p2, p2)
            
            G.add_edge(s1, s2, weight=score / 1000.0, source="STRING")
            edges_added += 1
    
    print(f"[PPI] STRING网络: {G.number_of_nodes():,} 节点, "
          f"{G.number_of_edges():,} 边 (score≥{combined_score_threshold})")
    
    # 保存网络
    nx.write_edgelist(G, DATA_PROCESSED / "ppi_string_edges.tsv", data=["weight"])
    
    # 网络统计
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "components": nx.number_connected_components(G),
        "largest_component_size": len(max(nx.connected_components(G), key=len)),
    }
    
    with open(DATA_PROCESSED / "ppi_stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"[PPI] 统计: {json.dumps(stats, indent=2)}")
    return G, alias_map


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1.5: PPI网络构建")
    print("=" * 60)
    G, alias_map = build_ppi_network()
    print(f"\n[PPI] 完成: {G.number_of_nodes():,} 节点, {G.number_of_edges():,} 边")
