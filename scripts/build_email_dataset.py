# -*- coding: utf-8 -*-
"""
build_email_dataset.py
- 从 data 目录（含多级子目录、文件无扩展名）批量提取邮件特征
- 标签来自 TREC06C 的 index 文件（如：'ham ../data/000/001'）
- 严格使用 spam->1, ham->0
- 输出时不包含 path 字段
"""

import os, sys, json, glob
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 将项目根目录加入 sys.path，保证能 import 到本包
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from email_feature_engine import parse_eml, extract_text_features, vectorize_feature_list  # 你已有的模块
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x): return x

# =================== 配置区 ===================
# 邮件所在根目录（你的 data 根）
INPUT_DIR    = r"E:\毕业设计\trec06c\data"
# TREC06C 的 index 文件（无扩展名也可）
LABEL_FILE   = r"E:\毕业设计\trec06c\full\index"

# 输出
OUTPUT_JSONL = r"E:\毕业设计\新测试\email_features.jsonl"
OUTPUT_NPY   = r"E:\毕业设计\新测试\email_features.npy"
BUCKET_SIZE  = 1024

# 扫描是否递归
RECURSIVE    = True
# =============================================

def _norm_path(p: str) -> str:
    """统一路径：去引号、反斜杠转斜杠、小写。"""
    p = (p or "").strip().strip('"').strip("'")
    return p.replace("\\", "/").lower()

def _abs_and_rel_keys(p_abs: Path, root_abs: Path) -> Tuple[str, str]:
    """返回规范化后的绝对路径键与“相对 root 的相对键”"""
    abs_k = _norm_path(str(p_abs))
    try:
        rel_k = _norm_path(str(p_abs.relative_to(root_abs)))
    except Exception:
        # 不在根目录下就退化为文件名
        rel_k = _norm_path(p_abs.name)
    return abs_k, rel_k

def load_labels_from_trec_index(index_file: str, dataset_root: str) -> Dict[str, Any]:
    """
    读取 TREC06C index：
        ham ../data/000/001
        spam ../data/000/002
    并建立两套键：绝对路径键、相对 INPUT_DIR 键
    """
    by_path: Dict[str, int] = {}

    idx_path = Path(index_file).resolve()
    base_for_rel = idx_path.parent  # 以 index 所在目录为相对路径基准
    root_abs = Path(dataset_root).resolve()

    spam_map = {"spam": 1, "ham": 0, "1": 1, "0": 0}

    with open(idx_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            raw_label, rel_path = parts[0], " ".join(parts[1:])

            raw_label_l = raw_label.lower()
            if raw_label_l not in spam_map:
                # 既不是 spam/ham 也不是 1/0，跳过
                continue
            lab = spam_map[raw_label_l]

            # 用 index 的目录作为相对路径基准来解析
            p_abs = (base_for_rel / rel_path).resolve()

            # 建立两个 key：绝对与相对（相对于 INPUT_DIR）
            abs_k, rel_k = _abs_and_rel_keys(p_abs, root_abs)
            by_path[abs_k] = lab
            by_path[rel_k] = lab

            # 再补充一个仅文件名键，防止极端场景（可选）
            by_path[_norm_path(p_abs.name)] = lab

    return {"by_path": by_path}

def iter_mail_files(root_dir: str, recursive: bool = True) -> List[str]:
    """
    递归收集“真实文件”（无扩展名也行），过滤掉目录。
    """
    root = Path(root_dir).resolve()
    # '**/*' 会把文件和目录都列出来，后面要过滤掉目录
    pats = ["**/*"] if recursive else ["*"]
    files: List[str] = []
    for pat in pats:
        for p in root.glob(pat):
            if p.is_file():
                files.append(str(p))
    return files

def main():
    # 基本路径检查
    in_root = Path(INPUT_DIR).resolve()
    if not in_root.exists():
        raise FileNotFoundError(f"未找到输入目录：{INPUT_DIR}")

    # 输出目录准备
    for out in (OUTPUT_JSONL, OUTPUT_NPY):
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)

    # 加载标签
    label_map = load_labels_from_trec_index(LABEL_FILE, dataset_root=str(in_root))
    by_path = label_map["by_path"]
    print(f"✅ 标签加载：路径映射 {len(by_path)} 条（spam→1, ham→0）")

    # 扫描文件
    files = iter_mail_files(str(in_root), RECURSIVE)
    print(f"📂 扫描目录: {in_root} (递归={RECURSIVE})，发现文件数：{len(files)}")

    if not files:
        print("❗ 未发现可解析的邮件文件，退出。")
        return

    feats = []
    miss, ok = 0, 0
    outf = open(OUTPUT_JSONL, "w", encoding="utf-8") if OUTPUT_JSONL else None

    for p in tqdm(files):
        try:
            # 跳过不可读或大小为 0 的文件
            try:
                if os.path.getsize(p) == 0:
                    continue
            except Exception:
                pass

            parsed = parse_eml(p)  # 你现有的解析函数（支持无扩展名）

            # —— 标签对齐：绝对/相对 两套键都尝试 —— #
            p_abs = Path(p).resolve()
            abs_k, rel_k = _abs_and_rel_keys(p_abs, in_root)

            label = None
            if abs_k in by_path:
                label = by_path[abs_k]
            elif rel_k in by_path:
                label = by_path[rel_k]
            else:
                # 再试文件名键（兜底）
                label = by_path.get(_norm_path(p_abs.name))

            feat = extract_text_features(parsed)  # 生成特征字典

            # —— 按需移除 path 字段 —— #
            if "path" in feat:
                del feat["path"]

            # 写入标签（未命中则 -1，便于你后续统计）
            feat["label"] = int(label) if label is not None else -1

            if label is None:
                miss += 1
            else:
                ok += 1

            feats.append(feat)
            if outf:
                outf.write(json.dumps(feat, ensure_ascii=False) + "\n")

        except Exception as e:
            # 解析失败也写一行错误信息（不含 path）
            if outf:
                outf.write(json.dumps({"error": f"parse_failed: {e}"}, ensure_ascii=False) + "\n")

    if outf:
        outf.close()
        print(f"📝 已写入特征：{OUTPUT_JSONL}（有效 {ok} 条，未匹配标签 {miss} 条，含错误行已记录）")

    # 向量化（仅当存在有效样本）
    valid = [x for x in feats if "subject_len" in x]
    if not valid:
        print("❗ 没有可向量化的有效特征，跳过。")
        return

    if OUTPUT_NPY:
        X, header = vectorize_feature_list(valid, bucket_size=BUCKET_SIZE)
        np.save(OUTPUT_NPY, X)
        print(f"🔢 已保存向量：{OUTPUT_NPY}，shape={X.shape}")

if __name__ == "__main__":
    main()
