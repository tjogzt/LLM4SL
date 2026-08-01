"""
步骤1.1 PubMed文献检索与下载
使用 Bio.Entrez API 搜索SL相关文献，下载摘要和元数据
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import time
import json
from pathlib import Path
from config import *

try:
    from Bio import Entrez
except ImportError:
    print("[Error] 需要 biopython: pip install biopython")
    sys.exit(1)

# 务必设置你的邮箱（NCBI要求）
Entrez.email = "taozhu@research.local"  # NCBI合规邮箱


def search_pubmed(term: str, max_results: int = PUBMED_MAX_RESULTS,
                  years: int = PUBMED_YEARS) -> list[str]:
    """搜索PubMed，返回PMID列表"""
    print(f"[PubMed] 搜索: {term[:80]}...")
    
    handle = Entrez.esearch(
        db="pubmed",
        term=term,
        retmax=max_results,
        datetype="pdat",
        mindate=f"{2025-years}/01/01",
        maxdate="2025/12/31",
        sort="relevance",
        usehistory="y"
    )
    results = Entrez.read(handle)
    handle.close()
    
    count = int(results["Count"])
    pmids = results["IdList"]
    webenv = results.get("WebEnv", "")
    query_key = results.get("QueryKey", "")
    
    print(f"[PubMed] 命中: {count} 篇, 下载前 {len(pmids)} 条")
    return pmids, webenv, query_key


def fetch_abstracts(pmids: list[str], batch_size: int = PUBMED_BATCH_SIZE,
                    output_dir: Path = DATA_RAW) -> list[dict]:
    """分批下载摘要和元数据"""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        print(f"[PubMed] 下载批次 {i//batch_size + 1}: {len(batch)} 条...")
        
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(batch),
                                   rettype="xml", retmode="xml")
            xml_data = handle.read()
            handle.close()
        except Exception as e:
            print(f"[PubMed] 批次 {i} 下载失败: {e}, 重试中...")
            time.sleep(5)
            handle = Entrez.efetch(db="pubmed", id=",".join(batch),
                                   rettype="xml", retmode="xml")
            xml_data = handle.read()
            handle.close()
        
        # 保存原始XML
        batch_file = output_dir / f"pubmed_batch_{i//batch_size:04d}.xml"
        with open(batch_file, "w", encoding="utf-8") as f:
            f.write(xml_data.decode("utf-8"))
        
        # 解析为JSON
        parsed = _parse_pubmed_xml(xml_data)
        records.extend(parsed)
        
        time.sleep(0.5)  # NCBI限速
    
    # 保存解析后的JSON
    json_file = DATA_PROCESSED / "pubmed_abstracts.json"
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print(f"[PubMed] 完成: {len(records)} 篇文章已保存到 {json_file}")
    return records


def _parse_pubmed_xml(xml_data: bytes) -> list[dict]:
    """解析PubMed XML为结构化字典"""
    from xml.etree import ElementTree as ET
    root = ET.fromstring(xml_data)
    records = []
    
    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = article.findtext(".//PMID", "")
            title = article.findtext(".//ArticleTitle", "")
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(
                (p.text or "") for p in abstract_parts if p.text
            )
            journal = article.findtext(".//Journal/Title", "")
            pub_date = article.findtext(".//PubDate/Year", "")
            
            records.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": pub_date,
            })
        except Exception as e:
            continue  # 跳过解析失败的条目
    
    return records


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1.1: PubMed文献检索")
    print("=" * 60)
    
    pmids, webenv, query_key = search_pubmed(PUBMED_SEARCH_TERM)
    records = fetch_abstracts(pmids)
    
    print(f"\n示例 (前3篇):")
    for r in records[:3]:
        print(f"  [{r['pmid']}] {r['title'][:80]}... ({r['journal']}, {r['year']})")
