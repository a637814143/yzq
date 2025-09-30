"""使用已训练好的 joblib 模型对邮件文本进行识别。

在脚本顶部配置 ``MODEL_PATH``、``VECTORIZER_PATH``（可选）和 ``INPUTS``
即可直接运行 ``python scripts/joblib_email_inference.py`` 完成预测。
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import joblib


# ===== 用户需根据自己的文件位置修改以下常量 =====
MODEL_PATH = Path(r"E:\毕业设计\新测试\spam_classifier_model.joblib")
# 若模型内部已包含特征提取环节，可保持 ``None``。
VECTORIZER_PATH: Path | None = None
# 可以填写若干文件或目录；留空时默认从标准输入读取。
INPUTS: Sequence[str] | None = [r"E:\毕业设计\邮件集\datacon2023-spoof-email-main\day1"]
# 读取邮件文本所用的编码。
ENCODING = "utf-8"
# 输出结果保存位置，使用 "-" 表示仅打印到控制台。
OUTPUT_PATH: str | Path = "E:\毕业设计\新测试"
# 仅处理以下扩展名的文件；设为 ``None`` 可禁用过滤。
ALLOWED_SUFFIXES: tuple[str, ...] | None = (".eml", ".txt", ".json", ".log", ".msg")
# ======== 常量配置结束 ========


def _load_serialized(path: Path | str):
    """通过 joblib（若可用）或 pickle 反序列化模型/向量器。"""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"未找到文件: {target}")

    try:
        return joblib.load(target)
    except Exception:
        with target.open("rb") as fh:
            return pickle.load(fh)


def _is_probably_text(path: Path, encoding: str, sample_size: int = 4096) -> bool:
    """粗略判断文件是否为文本内容。

    通过读取文件头部的少量字节尝试用给定编码解码；若包含 ``\x00`` 或
    解码失败，则认为是二进制文件，避免把 ``.npy`` 等特征文件当作邮件
    正文直接输出导致的“乱码”。
    """

    try:
        with path.open("rb") as fh:
            chunk = fh.read(sample_size)
    except OSError as exc:
        raise FileNotFoundError(f"无法读取文件: {path}") from exc

    if not chunk:
        return True

    if b"\x00" in chunk:
        return False

    try:
        chunk.decode(encoding)
        return True
    except UnicodeDecodeError:
        return False


def _should_process(path: Path, encoding: str) -> bool:
    if ALLOWED_SUFFIXES is not None and path.suffix.lower() not in ALLOWED_SUFFIXES:
        print(f"跳过不在允许扩展名列表中的文件: {path}", file=sys.stderr)
        return False

    if not _is_probably_text(path, encoding):
        print(f"跳过疑似二进制文件: {path}", file=sys.stderr)
        return False

    return True


def _collect_texts(paths: Sequence[str] | None, encoding: str) -> Tuple[List[str], List[str]]:
    """根据路径列表收集邮件正文。"""

    if not paths:
        text = sys.stdin.read()
        if not text:
            raise SystemExit("未提供输入数据")
        return [text], ["<stdin>"]

    texts: List[str] = []
    sources: List[str] = []

    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for file in sorted(path.rglob("*")):
                if file.is_file() and _should_process(file, encoding):
                    texts.append(file.read_text(encoding=encoding, errors="ignore"))
                    sources.append(str(file))
        elif path.is_file():
            if _should_process(path, encoding):
                texts.append(path.read_text(encoding=encoding, errors="ignore"))
                sources.append(str(path))
        else:
            raise FileNotFoundError(f"指定路径不存在: {raw}")

    if not texts:
        raise SystemExit("未从指定路径读取到任何可用的邮件文本")

    return texts, sources


def _predict(model, texts: Sequence[str], *, vectorizer=None) -> Tuple[Iterable, Iterable | None]:
    features = texts
    if vectorizer is not None:
        features = vectorizer.transform(texts)

    predictions = model.predict(features)

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(features)
        except Exception:
            proba = None

    return predictions, proba


def _format_results(
    sources: Sequence[str], predictions: Sequence, proba, classes: Sequence | None
) -> List[dict]:
    formatted: List[dict] = []
    for idx, (source, label) in enumerate(zip(sources, predictions)):
        entry = {"source": source, "prediction": _coerce(label)}
        if proba is not None:
            probs = proba[idx]
            labels = classes if classes is not None else range(len(probs))
            entry["probabilities"] = {
                _coerce(cls): float(prob) for cls, prob in zip(labels, probs)
            }
        formatted.append(entry)
    return formatted


def _coerce(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main() -> int:
    model = _load_serialized(MODEL_PATH)
    vectorizer = _load_serialized(VECTORIZER_PATH) if VECTORIZER_PATH else None

    texts, sources = _collect_texts(INPUTS, ENCODING)
    predictions, proba = _predict(model, texts, vectorizer=vectorizer)
    classes = getattr(model, "classes_", None)
    results = _format_results(sources, predictions, proba, classes)

    payload = json.dumps(results, ensure_ascii=False, indent=2)

    if OUTPUT_PATH == "-":
        print(payload)
    else:
        output_path = Path(OUTPUT_PATH)
        output_path.write_text(payload, encoding="utf-8")
        print(f"预测结果已保存至: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())