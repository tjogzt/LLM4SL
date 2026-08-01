"""
Phase 1 研究管线 - 全局配置
课题：LLM-GNN协同驱动的合成致死网络发现
"""

import os
from pathlib import Path

# === 项目根目录 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# === 目录结构 ===
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"           # PubMed原始下载
DATA_PROCESSED = DATA_DIR / "processed"  # 清洗后数据
DATA_EXTERNAL = DATA_DIR / "external"    # STRING/BioGRID/DepMap

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_MODELS = RESULTS_DIR / "models"
RESULTS_FIGURES = RESULTS_DIR / "figures"
RESULTS_TABLES = RESULTS_DIR / "tables"

# === LLM配置：DeepSeek-V4-Pro ===
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-v4-pro"  
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MAX_TOKENS = 8192
# 价格参考：¥1-2/百万tokens（输入），实际以官方定价为准

# === PubMed检索 ===
PUBMED_SEARCH_TERM = (
    '("synthetic lethal"[tiab] OR "synthetic lethality"[tiab] '
    'OR "genetic interaction"[tiab]) AND (cancer[tiab] OR tumor[tiab])'
)
PUBMED_YEARS = 5          # 近5年
PUBMED_MAX_RESULTS = 200000
PUBMED_BATCH_SIZE = 1000  # Entrez每次fetch上限

# === NER配置 ===
NER_TARGET_F1 = 0.85
NER_GENE_TYPES = ["gene", "protein"]  # 关注的实体类型

# === RE配置 ===
RE_CONFIDENCE_THRESHOLD = 0.8          # LLM高置信度阈值
RE_MIN_MENTIONS_CROSS_VALIDATION = 3   # 多源交叉验证阈值

# === 质控配置 ===
# 否定检测
NEGATION_TRIGGERS = [
    "not", "without", "absence", "lack of",
    "no", "neither", "failed to", "unable to",
    "independent of", "unrelated to"
]
NEGATION_WINDOW = 2  # ±2句上下文窗口

# 推测性语言
HEDGING_PATTERNS = [
    "may", "might", "potentially", "suggest",
    "could", "possibly", "appears to", "seems to",
    "putative", "candidate"
]
HEDGING_THRESHOLD = 2   # ≥2个推测词→降权
HEDGING_PENALTY = 0.5   # 置信度降权50%

# 来源分级权重
JOURNAL_WEIGHTS = {
    "top": 1.0,      # IF>10 (Nature/Cell/Science系)
    "high": 0.8,     # IF 5-10
    "low": 0.6,      # IF<5 或预印本
}

# === PPI网络配置 ===
STRING_COMBINED_SCORE_THRESHOLD = 0.7
PPI_NODE_COUNT = 18000    # 预期蛋白编码基因数
PPI_EDGE_COUNT = 300000   # 预期边数（过滤后）

# === GNN训练配置 ===
GNN_HIDDEN_DIM = 256
GNN_LAYERS = 3
ESM2_EMBED_DIM = 1280     # ESM-2 650M参数版本
TEXT_EMBED_DIM = 128      # BioBERT文本嵌入投影后维度
NODE_FEATURE_DIM = ESM2_EMBED_DIM + TEXT_EMBED_DIM  # 1408维
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
BATCH_SIZE = 1024

# 标签权重（弱监督训练）
LABEL_WEIGHTS = {
    "gold_positive": 1.0,      # SynLethDB金标准
    "llm_high_conf": 0.6,      # LLM高置信度提及
    "negative": 0.3,            # 负样本
}

# === AutoDL GPU配置 ===
AUTODL_GPU_TYPE = "A100-80G"
AUTODL_PRICE_PER_HOUR = 6.0   # ¥/小时（取中间值）

# === DepMap验证 ===
DEPMAP_VERSION = "26Q1"  # 预期版本
DEPMAP_CHRONOS_THRESHOLD = -0.5
DEPMAP_FDR_ALPHA = 0.05

print(f"[Config] 项目根目录: {PROJECT_ROOT}")
print(f"[Config] LLM模型: {DEEPSEEK_MODEL}")
print(f"[Config] GPU: {AUTODL_GPU_TYPE} @ ¥{AUTODL_PRICE_PER_HOUR}/h")
