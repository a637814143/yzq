# -*- coding: utf-8 -*-
"""
邮件文本特征提取模块。
输入：统一的 parsed email dict（见 parser.parse_eml）
输出：特征字典（可序列化为 JSON）
"""
from typing import Dict, Any
from .feature_utils import find_urls, count_short_links, find_emails, char_stats, top_k_ngrams
import re

RISK_KWS = {"免费","领取","限时","中奖","发票","点击","prize","win","free","urgent","invoice","verify"}

def extract_text_features(parsed: Dict) -> Dict[str, Any]:
    subj = parsed.get("subject","") or ""
    body = parsed.get("body","") or ""
    full = (subj + " " + body).strip()

    urls = find_urls(full)
    short_links = count_short_links(urls)
    emails = find_emails(full)
    subj_len = len(subj)
    total_len, upper_count, digit_count, punct_count = char_stats(full)
    body_len = len(body)
    words = re.findall(r"\w+", full)
    word_count = len(words)
    unique_word_count = len({w.lower() for w in words}) if words else 0
    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0.0
    uppercase_words = [w for w in words if len(w) > 1 and w.isupper()]
    uppercase_word_ratio = len(uppercase_words) / word_count if word_count else 0.0
    numeric_token_count = sum(1 for w in words if w.isdigit())
    # ratios
    total_len_for_ratio = max(1, total_len)
    upper_ratio = upper_count / total_len_for_ratio
    digit_ratio = digit_count / total_len_for_ratio
    punct_ratio = punct_count / total_len_for_ratio
    exclaim_count = full.count("!")
    question_count = full.count("?")
    money_symbol_count = sum(full.count(sym) for sym in ("$", "¥", "￥"))
    subj_exclaim = subj.count("!")
    subj_question = subj.count("?")
    subj_risk = int(any(kw in subj.lower() for kw in RISK_KWS))
    body_risk = int(any(kw in body.lower() for kw in RISK_KWS))
    total_risk_kw = sum(full.lower().count(kw) for kw in RISK_KWS)

    # url domains
    domain_set = set()
    for u in urls:
        host = re.sub(r"^https?://", "", u, flags=re.I).split("/")[0].lower()
        if host:
            domain_set.add(host)
    unique_url_domains = len(domain_set)

    # ngram summary (small)
    top_unigrams = top_k_ngrams(full, k=5, n=1)

    feat = {
        "path": parsed.get("path"),
        "subject_len": subj_len,#主题长度
        "subject_exclaim": subj_exclaim,#主题中感叹号数
        "subject_risk_kw": subj_risk,#主题中是否包含风险词汇
        "body_len": body_len,#正文的字符数
        "total_len": total_len,#邮件总字符数
        "upper_ratio": upper_ratio,#大小写字母占总字母数比例
        "digit_ratio": digit_ratio,#数字比例
        "punct_ratio": punct_ratio,#标点符号比例
        "exclaim_count": exclaim_count,#正文感叹号
        "question_count": question_count,#正文问号
        "money_symbol_count": money_symbol_count,#金额符号
        "word_count": word_count,#词汇数量
        "unique_word_ratio": (unique_word_count / word_count) if word_count else 0.0,#去重词汇占比
        "avg_word_len": avg_word_len,#平均词长
        "uppercase_word_ratio": uppercase_word_ratio,#大写词占比
        "numeric_token_count": numeric_token_count,#纯数字词数量
        "url_count": len(urls),#正文中URL数量
        "shortlink_count": short_links,#邮件正文中短连接的数量
        "unique_url_domains": unique_url_domains,#唯一URL域名数量
        "email_count": len(emails),#邮件正文中出现电子邮件地址的数量
        "attachments": parsed.get("attachments", 0),
        "is_html": 1 if bool(re.search(r"<[a-zA-Z]+[^>]*>", parsed.get("body","") or "")) else 0,
        "to_count": len(parsed.get("to", [])),
        "subject_question": subj_question,#主题中问号数量
        "subj_top_unigrams": top_unigrams,   # helpful for inspection
        "risk_kw_total": total_risk_kw,#风险关键词出现次数
    }
    # attach label if available
    if "label" in parsed:
        feat["label"] = parsed["label"]
    return feat
