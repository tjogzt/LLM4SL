"""
Phase 2.1: 图数据构建
将NER文本信号 + PPI网络 + SynLethDB标签 融合为PyG图数据
"""

import sys, json, pickle, re
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

def load_ppi_edges() -> tuple[pd.DataFrame, dict]:
    """加载STRING PPI并映射ENSP→gene symbol"""
    import gzip
    
    # 1. 构建映射
    ensp2symbol = {}
    alias_file = DATA_EXTERNAL / "string_human_v12.0_aliases.txt.gz"
    with gzip.open(alias_file, "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[2] == "Ensembl_HGNC_symbol":
                ensp2symbol[parts[0]] = parts[1]
    print(f"[Graph] ENSP→Symbol映射: {len(ensp2symbol):,}")
    
    # 2. 加载边并映射
    edges = []
    link_file = DATA_EXTERNAL / "string_human_v12.0_links.txt.gz"
    with gzip.open(link_file, "rt") as f:
        for line in f:
            if line.startswith("protein1"): continue
            p1, p2, score = line.strip().split()
            score = float(score)
            if score < STRING_COMBINED_SCORE_THRESHOLD * 1000:
                continue
            s1 = ensp2symbol.get(p1)
            s2 = ensp2symbol.get(p2)
            if s1 and s2:
                edges.append((s1, s2, score / 1000.0))
    
    df = pd.DataFrame(edges, columns=["gene_a", "gene_b", "weight"])
    print(f"[Graph] PPI边(score≥{STRING_COMBINED_SCORE_THRESHOLD}): {len(df):,} (映射率: {len(df)/236930*100:.0f}%)")
    return df, ensp2symbol

def load_ner_signals() -> dict:
    """从NER结果提取基因共现信号：在同一篇摘要中共同出现的基因对"""
    with open(DATA_PROCESSED / "ner_results.jsonl") as f:
        records = [json.loads(l) for l in f]
    
    co_mentions = defaultdict(lambda: {"count": 0, "pmids": set(), "contexts": []})
    
    for rec in records:
        genes = rec.get("genes", [])
        if genes and isinstance(genes[0], str):
            symbols = genes
            contexts = [""] * len(genes)
        else:
            symbols = [g.get("symbol", str(g)) if isinstance(g, dict) else str(g) for g in genes]
            contexts = [g.get("context", "") if isinstance(g, dict) else "" for g in genes]
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                pair = tuple(sorted([symbols[i], symbols[j]]))
                co_mentions[pair]["count"] += 1
                co_mentions[pair]["pmids"].add(rec["pmid"])
                if contexts[i] == "mutation" or contexts[j] == "mutation":
                    co_mentions[pair]["contexts"].append("mutation")
    
    print(f"[Graph] NER共现基因对: {len(co_mentions):,}")
    mult_mention = sum(1 for v in co_mentions.values() if v["count"] >= 3)
    print(f"[Graph] ≥3次共现: {mult_mention:,}")
    return dict(co_mentions)

def load_synlethdb() -> tuple[set, set, dict]:
    """加载SynLethDB正负样本"""
    positives = set()
    negatives = set()
    sl_details = {}
    
    # 正样本
    sl_file = DATA_EXTERNAL / "Human.SL.detailed.tsv"
    if sl_file.exists():
        df = pd.read_csv(sl_file, sep="\t")
        for _, row in df.iterrows():
            a, b = str(row["x_name"]).strip(), str(row["y_name"]).strip()
            if a and b and a != "nan" and b != "nan":
                positives.add(tuple(sorted([a, b])))
                sl_details[tuple(sorted([a, b]))] = {
                    "cancer": row.get("cancer", ""),
                    "pubmed_id": row.get("pubmed_id", ""),
                }
    
    # 负样本
    non_file = DATA_EXTERNAL / "Human.non.SL.detailed.tsv"
    if non_file.exists():
        df = pd.read_csv(non_file, sep="\t")
        for _, row in df.iterrows():
            a, b = str(row["x_name"]).strip(), str(row["y_name"]).strip()
            if a and b and a != "nan" and b != "nan":
                negatives.add(tuple(sorted([a, b])))
    
    print(f"[Graph] SynLethDB: {len(positives):,}正 / {len(negatives):,}负")
    return positives, negatives, sl_details

def build_graph() -> tuple[dict, dict]:
    """构建统一的图结构"""
    print("=" * 50)
    print("Phase 2.1: 图数据构建")
    print("=" * 50)
    
    ppi, ensp_map = load_ppi_edges()
    ner = load_ner_signals()
    sl_pos, sl_neg, sl_details = load_synlethdb()
    
    # === 节点映射 ===
    all_genes = set(ppi["gene_a"]) | set(ppi["gene_b"])
    ner_genes = set()
    for a, b in ner:
        ner_genes.add(a)
        ner_genes.add(b)
    
    print(f"\nPPI基因: {len(all_genes):,} | NER提及基因: {len(ner_genes):,}")
    print(f"交集: {len(all_genes & ner_genes):,}")
    
    # 节点ID映射（只保留在PPI中的基因）
    gene_list = sorted(all_genes)
    gene2id = {g: i for i, g in enumerate(gene_list)}
    
    # === 边列表 ===
    edge_index = []
    edge_weight = []
    for _, row in ppi.iterrows():
        a, b, w = row["gene_a"], row["gene_b"], row["weight"]
        if a in gene2id and b in gene2id:
            edge_index.append([gene2id[a], gene2id[b]])
            edge_index.append([gene2id[b], gene2id[a]])  # 无向图
            edge_weight.extend([w, w])
    
    edge_index = np.array(edge_index).T  # [2, num_edges]
    edge_weight = np.array(edge_weight)
    
    print(f"图: {len(gene2id):,}节点, {edge_index.shape[1]:,}边")
    
    # === 文本信号特征 ===
    # 每个节点：NER提及次数 + 含突变背景的次数
    text_features = np.zeros((len(gene_list), 4))
    gene_ner_count = defaultdict(int)
    gene_mutation_count = defaultdict(int)
    gene_pmids = defaultdict(set)
    
    for (a, b), info in ner.items():
        gene_ner_count[a] += info["count"]
        gene_ner_count[b] += info["count"]
        gene_pmids[a].update(info["pmids"])
        gene_pmids[b].update(info["pmids"])
        if "mutation" in info.get("contexts", []):
            gene_mutation_count[a] += 1
            gene_mutation_count[b] += 1
    
    for g, idx in gene2id.items():
        text_features[idx, 0] = gene_ner_count.get(g, 0)       # NER共现次数
        text_features[idx, 1] = gene_mutation_count.get(g, 0)   # 突变上下文次数
        text_features[idx, 2] = len(gene_pmids.get(g, set()))   # 提及该基因的文章数
        text_features[idx, 3] = int(g in ner_genes)             # 是否在NER中出现
    
    # 归一化
    text_features = (text_features - text_features.mean(0)) / (text_features.std(0) + 1e-8)
    
    # === 训练标签 ===
    # 1=正样本(SynLethDB), 0=负样本, -1=未知
    labels = np.full(len(gene_list), -1, dtype=np.float32)
    
    # 对每对SL基因标注正/负标签（在边级别标注，节点级别用于初始化）
    pos_set = sl_pos
    neg_set = sl_neg
    
    # === 构建边标签（用于链接预测） ===
    sl_pair_labels = {}  # (gene_a_idx, gene_b_idx) -> label
    all_ppi_pairs = set()
    for i in range(edge_index.shape[1] // 2):
        a, b = edge_index[0, 2*i], edge_index[1, 2*i]
        ga, gb = gene_list[a], gene_list[b]
        pair = tuple(sorted([ga, gb]))
        all_ppi_pairs.add(pair)
        if pair in pos_set:
            sl_pair_labels[(a, b)] = 1
            sl_pair_labels[(b, a)] = 1
        elif pair in neg_set:
            sl_pair_labels[(a, b)] = 0
            sl_pair_labels[(b, a)] = 0
    
    pos_in_ppi = sum(1 for p in pos_set if p in all_ppi_pairs)
    neg_in_ppi = sum(1 for p in neg_set if p in all_ppi_pairs)
    print(f"PPI中的SL正样本: {pos_in_ppi:,} / {len(pos_set):,}")
    print(f"PPI中的SL负样本: {neg_in_ppi:,} / {len(neg_set):,}")
    
    # === NER弱监督边权重 ===
    ner_edge_weights = {}
    for (a, b), info in ner.items():
        if a in gene2id and b in gene2id:
            ia, ib = gene2id[a], gene2id[b]
            confidence = min(1.0, info["count"] / RE_MIN_MENTIONS_CROSS_VALIDATION)
            ner_edge_weights[(ia, ib)] = confidence * LABEL_WEIGHTS["llm_high_conf"]
            ner_edge_weights[(ib, ia)] = confidence * LABEL_WEIGHTS["llm_high_conf"]
    
    # === 保存 ===
    graph_data = {
        "gene_list": gene_list,
        "gene2id": gene2id,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "text_features": text_features,
        "sl_pair_labels": sl_pair_labels,
        "ner_edge_weights": ner_edge_weights,
        "pos_in_ppi": pos_in_ppi,
        "neg_in_ppi": neg_in_ppi,
    }
    
    output = DATA_PROCESSED / "graph_data.pkl"
    with open(output, "wb") as f:
        pickle.dump(graph_data, f)
    print(f"\n✅ 图数据已保存: {output}")
    print(f"   节点: {len(gene_list):,} | 边: {edge_index.shape[1]:,}")
    print(f"   正标签: {pos_in_ppi:,} | 负标签: {neg_in_ppi:,}")
    print(f"   NER边权重: {len(ner_edge_weights):,}")
    
    return graph_data

if __name__ == "__main__":
    build_graph()
