"""使用训练好的模型对多种格式的邮件文件进行预测。

脚本提供 ``predict_emails`` 函数，可在其它 Python 代码中直接调用，
无需依赖命令行参数解析。函数会重用 ``email_feature_engine`` 包中已经
实现的解析与特征提取流程，从原始邮件文本构造与训练阶段一致的
特征向量，再交由 ``joblib`` 保存的模型进行推理。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import joblib

from email_feature_engine import (
    BUCKET_SIZE,
    parse_csv_row,
    extract_text_features,
    parse_eml,
    parse_json,
    vectorize_feature_list,
)


DEFAULT_MODEL_PATH = Path("E:\毕业设计\新测试\朴素贝叶斯\\nb_model.joblib")
# 解析不同邮件类型所使用的函数映射。
Parser = Callable[[Path], List[dict]]


def _parse_eml_file(path: Path) -> List[dict]:
    return [parse_eml(str(path))]


def _parse_json_file(path: Path) -> List[dict]:
    return [parse_json(str(path))]


def _parse_csv_file(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        parsed_rows: List[dict] = []
        for idx, row in enumerate(reader, start=1):
            parsed = parse_csv_row(row)
            parsed.setdefault("path", f"{path}#row={idx}")
            parsed_rows.append(parsed)

    if not parsed_rows:
        raise ValueError("CSV 文件未包含任何数据行")

    return parsed_rows


def _parse_txt_file(path: Path) -> List[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")

    payload = {
        "path": str(path),
        "subject": "",
        "from": "",
        "to": [],
        "body": text,
        "raw": text,
        "attachments": 0,
    }

    return [payload]


PARSER_REGISTRY: Dict[str, Parser] = {
    ".eml": _parse_eml_file,
    ".json": _parse_json_file,
    ".csv": _parse_csv_file,
    ".txt": _parse_txt_file,
}

DEFAULT_ALLOWED_SUFFIXES = tuple(PARSER_REGISTRY.keys())
# 根据用户实际环境配置默认的输入目录或文件
DEFAULT_INPUT_PATHS: Tuple[Path, ...] = (
    Path(r"E:\毕业设计\邮件集\datacon2023-spoof-email-main\day1"),
)


def _iter_input_paths(inputs: Sequence[str]) -> Iterable[Path]:
    """Expand ``inputs`` into a list of files.

    ``inputs`` 可以包含单个文件或目录；目录会被递归遍历以收集其中所有
    文件。若路径不存在将抛出 ``FileNotFoundError``。
    """

    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for file in sorted(path.rglob("*")):
                if file.is_file():
                    yield file
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(f"指定路径不存在: {raw}")


def _should_process(path: Path, allowed_suffixes: Tuple[str, ...] | None) -> bool:
    if allowed_suffixes is None:
        return True
    return path.suffix.lower() in allowed_suffixes


def _resolve_parser(path: Path) -> Parser:
    suffix = path.suffix.lower()
    parser = PARSER_REGISTRY.get(suffix)
    if parser is None:
        raise ValueError(f"暂不支持的邮件文件类型: {suffix or '<无扩展名>'}")
    return parser


def _collect_features(paths: Sequence[Path]) -> Tuple[List[str], List[dict], List[str]]:
    """Parse邮件并提取特征。

    返回值包含三部分：

    - 成功处理的源文件标识 ``str`` 列表；
    - 与每个源文件对应的特征 ``dict``；
    - 处理失败的文件路径（用于告警）。
    """

    processed: List[str] = []
    features: List[dict] = []
    failures: List[str] = []

    for path in paths:
        try:
            parser = _resolve_parser(path)
            parsed_payloads = parser(path)
        except Exception as exc:  # pragma: no cover - 仅用于错误提示
            failures.append(f"{path}: {exc}")
            continue

        for payload in parsed_payloads:
            try:
                source = str(payload.get("path") or path)
                normalized = dict(payload)
                normalized["path"] = source
                feat = extract_text_features(normalized)
                # 保留原始路径便于溯源；向量化时会忽略该字段。
                feat["path"] = source
                features.append(feat)
                processed.append(source)
            except Exception as exc:  # pragma: no cover - 仅用于错误提示
                failures.append(f"{source}: {exc}")

    if failures:
        print(
            "警告：部分邮件解析失败，已跳过：",
            *failures,
            sep="\n",
            file=sys.stderr,
        )

    if not features:
        raise SystemExit("未能从输入文件中提取到任何特征")

    return processed, features, failures


def _vectorize(features: Sequence[dict], bucket_size: int):
    """将特征 ``dict`` 转换为模型可接受的 numpy 数组。"""

    matrix, _ = vectorize_feature_list(list(features), bucket_size=bucket_size)
    return matrix


def _load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"未找到模型文件: {path}")
    return joblib.load(path)


def _predict(model, features_matrix):
    predictions = model.predict(features_matrix)
    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(features_matrix)
        except Exception:  # pragma: no cover - 个别模型可能不支持
            probabilities = None
    return predictions, probabilities


def _format_results(
    sources: Sequence[str],
    predictions,
    probabilities,
    classes: Sequence | None,
):
    results = []
    for idx, (source, label) in enumerate(zip(sources, predictions)):
        entry = {
            "source": source,
            "prediction": int(label) if isinstance(label, (bool, int, float)) else label,
        }
        if probabilities is not None:
            probs = probabilities[idx]
            labels = classes if classes is not None else range(len(probs))
            entry["probabilities"] = {
                str(cls): float(prob) for cls, prob in zip(labels, probs)
            }
        results.append(entry)
    return results


def predict_emails(
    inputs: Sequence[str | Path] | None = None,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    bucket_size: int = BUCKET_SIZE,
    allowed_suffixes: Sequence[str] | None = DEFAULT_ALLOWED_SUFFIXES,
    output_path: str | Path | None = None,
    emit_console: bool = True,
) -> List[dict]:
    """识别 ``inputs`` 中的邮件并返回结果列表。

    参数
    ----
    inputs:
        待识别的邮件文件或目录，可混合传入；目录会被递归遍历。
    model_path:
        训练好的 ``joblib`` 模型路径。
    bucket_size:
        特征向量化时使用的哈希桶大小，应与训练阶段保持一致。
    allowed_suffixes:
        允许处理的文件扩展名序列；传入 ``None`` 表示不过滤。
    output_path:
        若提供，则会将预测结果写入对应的 JSON 文件。
    emit_console:
        为 ``True`` 时会在控制台打印可读的预测信息。

    返回
    ----
    list of dict
        每封邮件对应一个结果字典，包含 ``source``、``prediction``，如模型
        支持概率输出还会包含 ``probabilities`` 字段。
    """

    if inputs is None:
        inputs = DEFAULT_INPUT_PATHS

    if not inputs:
        raise ValueError("inputs 不能为空")

    path_inputs = list(inputs)

    model = _load_model(Path(model_path))

    if allowed_suffixes is not None:
        normalized_suffixes: Tuple[str, ...] | None = tuple(
            suffix.lower() for suffix in allowed_suffixes
        )
    else:
        normalized_suffixes = None

    candidates = list(_iter_input_paths(path_inputs))
    targets = [p for p in candidates if _should_process(p, normalized_suffixes)]

    if not targets:
        raise FileNotFoundError("未找到任何匹配的邮件文件")

    sources, features, _ = _collect_features(targets)
    matrix = _vectorize(features, bucket_size=bucket_size)

    predictions, probabilities = _predict(model, matrix)
    classes = getattr(model, "classes_", None)
    results = _format_results(sources, predictions, probabilities, classes)

    if emit_console:
        for item in results:
            label = item["prediction"]
            human_readable = (
                "垃圾邮件" if str(label) in {"1", "True"} or label == 1 else "正常邮件"
            )
            print(f"{item['source']}: {human_readable}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if emit_console:
            print(f"预测结果已保存至: {output_path}")

    return results


if __name__ == "__main__":
    predict_emails()