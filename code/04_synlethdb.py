"""下载SynLethDB 3.0 SL金标准标签数据"""
import sys, json, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import *

print("[SynLethDB] 下载SL金标准数据...")

# SynLethDB 3.0 提供JSON API
url = "http://synlethdb.sist.shanghaitech.edu.cn/api/v1/sl_pairs"
try:
    resp = requests.get(url, timeout=30, params={"species": "human", "limit": 100000})
    
    if resp.status_code == 200:
        data = resp.json()
        pairs = data.get("data", data) if isinstance(data, dict) else data
        
        output = DATA_EXTERNAL / "synlethdb_sl_pairs.json"
        with open(output, "w") as f:
            json.dump(pairs, f, ensure_ascii=False)
        
        print(f"[SynLethDB] 已保存 {len(pairs) if isinstance(pairs, list) else '?'} 条SL对到 {output}")
    else:
        print(f"[SynLethDB] API返回 {resp.status_code}，尝试本地缓存...")
        
except Exception as e:
    print(f"[SynLethDB] API不可用: {e}")
    print("[SynLethDB] 将在本机创建占位文件，后续手动下载")
    
    # 创建占位
    placeholder = {
        "source": "SynLethDB 3.0",
        "url": "http://synlethdb.sist.shanghaitech.edu.cn/",
        "note": "请手动下载SL对数据，或使用API获取",
        "expected_count": 51411
    }
    with open(DATA_EXTERNAL / "synlethdb_placeholder.json", "w") as f:
        json.dump(placeholder, f, ensure_ascii=False, indent=2)

print("[SynLethDB] 完成")
