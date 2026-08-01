"""
步骤1.2 DeepSeek-V4-Pro NER：从PubMed摘要中识别基因实体
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import json
import time
from pathlib import Path
from config import *

try:
    from openai import OpenAI
except ImportError:
    print("[Error] 需要 openai: pip install openai")
    sys.exit(1)

# DeepSeek API兼容OpenAI SDK
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

NER_SYSTEM_PROMPT = """你是一个生物医学命名实体识别系统。从给定的PubMed摘要中识别所有基因和蛋白质名称。

要求：
1. 仅识别人类基因/蛋白质名称（HGNC标准符号）
2. 如果摘要中提及了基因别名，请同时输出标准符号和别名
3. 如果基因名称在特定突变背景（如BRCA1突变、TP53缺失）中被提及，请标注该背景
4. 区分基因名称和化学物质/药物名称（不要标注后者）

输出JSON格式：
{
  "genes": [
    {
      "symbol": "BRCA1",
      "aliases": [],
      "context": "mutation",
      "sentence": "BRCA1-mutated tumors showed sensitivity to PARP inhibition"
    }
  ]
}"""


def ner_single_abstract(abstract: str, pmid: str,
                        model: str = DEEPSEEK_MODEL) -> dict:
    """对单篇摘要做NER"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": NER_SYSTEM_PROMPT},
                {"role": "user", "content": f"PMID: {pmid}\n\n摘要:\n{abstract}"}
            ],
            temperature=0.0,      # 确定性输出
            max_tokens=DEEPSEEK_MAX_TOKENS,
            response_format={"type": "json_object"}  # 严格JSON输出
        )
        result = json.loads(response.choices[0].message.content)
        result["pmid"] = pmid
        return result
    except Exception as e:
        print(f"[NER] PMID {pmid} 失败: {e}")
        return {"pmid": pmid, "genes": [], "error": str(e)}


def ner_batch(records: list[dict], start_idx: int = 0,
              model: str = DEEPSEEK_MODEL,
              output_file: Path = DATA_PROCESSED / "ner_results.jsonl") -> dict:
    """批量NER，结果写入JSONL"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    total_tokens = 0
    
    with open(output_file, "a", encoding="utf-8") as f:
        for i, record in enumerate(records[start_idx:], start=start_idx):
            pmid = record.get("pmid", "")
            abstract = record.get("abstract", "")
            
            if not abstract or len(abstract) < 50:
                continue  # 跳过空摘要
            
            print(f"[NER] {i+1}/{len(records)}: PMID {pmid} "
                  f"({len(abstract)}字符)")
            
            ner_result = ner_single_abstract(abstract, pmid, model=model)
            results[pmid] = ner_result
            
            f.write(json.dumps(ner_result, ensure_ascii=False) + "\n")
            f.flush()
            
            # 估算token消耗
            total_tokens += len(abstract) // 2  # 粗略估计
            
            time.sleep(0.2)  # API限速
            
            # 每100条打印预算
            if (i + 1) % 100 == 0:
                cost_est = total_tokens / 1_000_000 * 1.5  # ¥1.5/M tokens
                print(f"  [预算] 已处理 {i+1} 篇, "
                      f"估算 token: {total_tokens:,}, 估算费用: ¥{cost_est:.2f}")
    
    # 保存汇总
    summary = {
        "total_processed": len(results),
        "total_tokens_est": total_tokens,
        "total_cost_est_rmb": total_tokens / 1_000_000 * 1.5,
        "model": model,
    }
    with open(DATA_PROCESSED / "ner_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n[NER] 完成: {len(results)} 篇文章")
    print(f"[NER] 估算总费用: ¥{summary['total_cost_est_rmb']:.2f}")
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1.2: DeepSeek-V4-Pro NER")
    print("=" * 60)
    
    # 加载PubMed摘要
    pubmed_file = DATA_PROCESSED / "pubmed_abstracts.json"
    if not pubmed_file.exists():
        print(f"[Error] 先运行 01_pubmed_search.py 下载文献")
        sys.exit(1)
    
    with open(pubmed_file) as f:
        records = json.load(f)
    
    print(f"[NER] 已加载 {len(records)} 篇文章")
    print(f"[NER] 模型: {DEEPSEEK_MODEL}")
    print(f"[NER] 预估费用: ¥{len(records) * 500 / 1_000_000 * 1.5:.2f} "
          f"(按每篇500 tokens估算)")
    print()
    
    # 先测试5篇
    print("[NER] 先测试5篇...")
    test_results = ner_batch(records[:5], output_file=DATA_PROCESSED / "ner_results_test.jsonl")
    
    # 人工检查质量
    for pmid, r in list(test_results.items())[:2]:
        genes = r.get("genes", [])
        print(f"\n  PMID {pmid}: 识别到 {len(genes)} 个基因")
        for g in genes[:3]:
            print(f"    - {g.get('symbol', '?')}"
                  f"{' (别名: ' + ', '.join(g.get('aliases', [])) + ')' if g.get('aliases') else ''}"
                  f" [{g.get('context', '')}]")
    
    ans = input("\n质量OK吗? 继续全部处理? (y/n): ")
    if ans.lower() == 'y':
        ner_batch(records[5:], start_idx=5,
                  output_file=DATA_PROCESSED / "ner_results.jsonl")
