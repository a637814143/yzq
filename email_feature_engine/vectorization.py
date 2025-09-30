# -*- coding: utf-8 -*-
"""
把特征 dict -> 固定长度向量。
策略：
- 数值特征以原值/比例放入向量
- 部分类别特征（如 subject unigram top-k）使用 stable_hash -> bucket index
- 输出 vector (numpy array) 与 feature order metadata
"""
import numpy as np
from typing import Dict, List
from .feature_utils import stable_hash

NUMERIC_FIELDS = [
    "subject_len","subject_exclaim","subject_risk_kw",
    "body_len","total_len","upper_ratio","digit_ratio",
    "punct_ratio","exclaim_count","url_count","shortlink_count",
    "email_count","attachments","is_html","to_count"
]

# hash bucket settings for textual tokens
BUCKET_SIZE = 1024

def _vectorize_one(feat: Dict, bucket_size: int = BUCKET_SIZE) -> np.ndarray:
    vec = []
    for f in NUMERIC_FIELDS:
        vec.append(float(feat.get(f, 0.0)))
    # textual hashed buckets: take top unigrams if exist
    bucket = [0.0] * bucket_size
    tu = feat.get("subj_top_unigrams", [])
    for token, cnt in tu:
        idx = stable_hash(token, bits=16) % bucket_size
        bucket[idx] += float(cnt)
    vec.extend(bucket)
    return np.array(vec, dtype=np.float32)

def vectorize_feature_list(feature_list: List[Dict], bucket_size: int = BUCKET_SIZE) -> (np.ndarray, List[str]):
    X = []
    for f in feature_list:
        X.append(_vectorize_one(f, bucket_size=bucket_size))
    X = np.stack(X, axis=0)
    # return also header (feature names) for interpretability
    header = NUMERIC_FIELDS + [f"hash_{i}" for i in range(bucket_size)]
    return X, header
