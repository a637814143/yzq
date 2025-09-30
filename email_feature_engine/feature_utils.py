# -*- coding: utf-8 -*-
"""
基础文本特征/工具集 (email text utilities)
提供：
- URL/EMAIL 正则、短链检测
- 文本统计（upper/digit/punct ratios）
- ngram tokenizer util（可选）
- safe_hash 用于类别哈希化
"""
import re
import hashlib
from collections import Counter
from typing import List, Tuple

URL_RE = re.compile(r"https?://[^\s'\"<>()]+", re.I)
WWW_RE = re.compile(r"www\.[^\s'\"<>()]+", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.I)
SHORTENERS = {"bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly", "is.gd", "buff.ly"}

PUNCT_SET = set(".,;:!?\"'()[]{}<>-—…，。；：！？“”‘’（）【】《》")

def find_urls(text: str) -> List[str]:
    if not text:
        return []
    urls = URL_RE.findall(text) + WWW_RE.findall(text)
    # normalize
    return urls

def count_short_links(urls: List[str]) -> int:
    cnt = 0
    for u in urls:
        try:
            host = re.sub(r"^https?://", "", u, flags=re.I).split("/")[0].lower()
            if any(host.endswith(s) or host == s for s in SHORTENERS):
                cnt += 1
        except Exception:
            continue
    return cnt

def find_emails(text: str) -> List[str]:
    if not text:
        return []
    return EMAIL_RE.findall(text)

def char_stats(text: str) -> Tuple[int,int,int,int]:
    """返回 (length, upper_count, digit_count, punct_count)"""
    if not text:
        return 0,0,0,0
    length = len(text)
    upper = sum(1 for c in text if 'A' <= c <= 'Z')
    digit = sum(1 for c in text if c.isdigit())
    punct = sum(1 for c in text if c in PUNCT_SET)
    return length, upper, digit, punct

def stable_hash(s: str, bits: int = 32) -> int:
    """稳定哈希，返回非负 int（适合做 feature hashing index）"""
    if s is None:
        s = ""
    h = hashlib.md5(s.encode('utf-8', errors='ignore')).hexdigest()
    return int(h, 16) % (2**bits)

def top_k_ngrams(text: str, k=10, n=1) -> List[Tuple[str,int]]:
    """简单 n-gram 统计（空格分词或字符级）"""
    if not text:
        return []
    words = re.findall(r"\w+", text.lower())
    if n == 1:
        ctr = Counter(words)
    else:
        ngrams = []
        for i in range(len(words)-n+1):
            ngrams.append(" ".join(words[i:i+n]))
        ctr = Counter(ngrams)
    return ctr.most_common(k)
