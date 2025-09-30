# -*- coding: utf-8 -*-
"""
邮件解析器：将三类输入（.eml / json / csv）解析成统一 dict

输出 dict 模板：
{
  "path": "...",        # 文件路径或来源id
  "subject": "...",
  "from": "...",
  "to": ["a@x", "b@y"],
  "body": "...",        # 优先 text/plain，回退 text/html -> text
  "raw": "...",         # 原始文本（可选）
  "attachments": N      # 如可得
}
"""
import os
import json
from email import policy
from email.parser import BytesParser
from email.utils import getaddresses
from typing import Dict, Optional, List
import html as ihtml
import re
import csv
from .feature_utils import EMAIL_RE

def _html_to_text(html: str) -> str:
    if not isinstance(html, str):
        return ""
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<\s*br\s*/?>", "\n", html)
    text = re.sub(r"(?s)<.*?>", " ", html)
    text = ihtml.unescape(text)
    return text.strip()

def parse_eml(path: str) -> Dict:
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    subject = str(msg.get("subject") or "")
    from_field = str(msg.get("from") or "")
    to_list = []
    for h in ("to","cc","bcc"):
        to_list += [addr for name, addr in getaddresses(msg.get_all(h, []))]
    # body extract
    body_parts = []
    if msg.is_multipart():
        # prefer text/plain
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body_parts.append(part.get_content())
                except Exception:
                    pass
        if not body_parts:
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    try:
                        body_parts.append(_html_to_text(part.get_content()))
                    except Exception:
                        pass
    else:
        c = msg.get_content_type()
        try:
            if c == "text/plain":
                body_parts.append(msg.get_content())
            elif c == "text/html":
                body_parts.append(_html_to_text(msg.get_content()))
        except Exception:
            pass
    body = " ".join([b for b in body_parts if isinstance(b,str)])
    # attachments count (heuristic)
    attach_count = 0
    if msg.is_multipart():
        for part in msg.walk():
            disp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disp or part.get_filename():
                attach_count += 1
    return {
        "path": path,
        "subject": subject,
        "from": from_field,
        "to": to_list,
        "body": body,
        "raw": None,
        "attachments": attach_count
    }

def parse_json(path: str, body_key: str = "body", subject_key: str = "subject", label_key: Optional[str]=None) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return {
        "path": path,
        "subject": j.get(subject_key, ""),
        "from": j.get("from", ""),
        "to": j.get("to", []),
        "body": j.get(body_key, ""),
        "raw": j,
        "attachments": len(j.get("attachments", [])) if isinstance(j.get("attachments", []), list) else 0,
        **({ "label": j.get(label_key) } if label_key else {})
    }

def parse_csv_row(row: Dict, subject_col="subject", body_col="body", from_col="from", label_col=None) -> Dict:
    subj = row.get(subject_col, "")
    body = row.get(body_col, "")
    frm = row.get(from_col, "")
    out = {"path": None, "subject": subj, "from": frm, "to": [], "body": body, "raw": row, "attachments": 0}
    if label_col and label_col in row:
        out["label"] = row[label_col]
    return out
