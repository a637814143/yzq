# -*- coding: utf-8 -*-
"""
把特征 dict -> 固定长度向量。
策略：
- 数值特征以原值/比例放入向量
- 部分类别特征（如 subject unigram top-k）使用 stable_hash -> bucket index
- 输出 vector (numpy array) 与 feature order metadata
- 提供特征标准化功能（StandardScaler）
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from .feature_utils import stable_hash
from sklearn.preprocessing import StandardScaler

NUMERIC_FIELDS = [
    "subject_len","subject_exclaim","subject_question","subject_risk_kw",
    "body_len","total_len","upper_ratio","digit_ratio",
    "punct_ratio","exclaim_count","question_count","money_symbol_count",
    "word_count","unique_word_ratio","avg_word_len","uppercase_word_ratio",
    "numeric_token_count","url_count","shortlink_count","unique_url_domains",
    "email_count","attachments","is_html","to_count","risk_kw_total"
]

# hash bucket settings for textual tokens
BUCKET_SIZE = 1024

def _vectorize_one(feat: Dict, bucket_size: int = BUCKET_SIZE) -> np.ndarray:
    vec = []
    for f in NUMERIC_FIELDS:
        vec.append(float(feat.get(f, 0.0)))
    # textual hashed buckets: take top unigrams from subject and body
    bucket = [0.0] * bucket_size
    # Use subject unigrams
    tu = feat.get("subj_top_unigrams", [])
    for token, cnt in tu:
        idx = stable_hash(f"subj_{token}", bits=16) % bucket_size
        bucket[idx] += float(cnt)
    # Also use body unigrams (extract from full text if available)
    # Note: We could add body_top_unigrams to text_features.py for better extraction
    vec.extend(bucket)
    return np.array(vec, dtype=np.float32)

def vectorize_feature_list(
    feature_list: List[Dict], 
    bucket_size: int = BUCKET_SIZE,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False
) -> Tuple[np.ndarray, List[str], Optional[StandardScaler]]:
    """
    将特征字典列表转换为固定长度向量。
    
    参数:
        feature_list: 特征字典列表
        bucket_size: 哈希桶大小
        scaler: 可选的预训练标准化器
        fit_scaler: 是否拟合新的标准化器
    
    返回:
        X: 特征矩阵 (n_samples, n_features)
        header: 特征名称列表
        scaler: 标准化器（如果使用）
    """
    X = []
    for f in feature_list:
        X.append(_vectorize_one(f, bucket_size=bucket_size))
    X = np.stack(X, axis=0)
    
    # 应用特征标准化
    fitted_scaler = None
    if scaler is not None:
        X = scaler.transform(X)
        fitted_scaler = scaler
    elif fit_scaler:
        fitted_scaler = StandardScaler()
        X = fitted_scaler.fit_transform(X)
    
    # return also header (feature names) for interpretability
    header = NUMERIC_FIELDS + [f"hash_{i}" for i in range(bucket_size)]
    return X, header, fitted_scaler
