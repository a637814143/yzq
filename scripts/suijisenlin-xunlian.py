"""使用随机森林算法训练垃圾邮件分类模型的脚本。

本脚本与 ``SVM-xunlian.py`` 的输入输出风格保持一致：

- 读取 ``.npy`` 特征文件与 ``.txt`` 标签文件（spam/ham）
- 默认按照 70%/15%/15% 划分训练、验证、测试集
- 输出与 SVM 脚本一致的中文评估指标与混淆矩阵格式
- 训练完成后将模型保存为 ``.joblib`` 文件
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 默认路径，可通过环境变量覆盖
DEFAULT_FEATURES_PATH = Path(
    os.environ.get("RF_FEATURES_PATH", r"E:\\毕业设计\\新测试\\新的\\email_features.npy")
)
DEFAULT_LABELS_PATH = Path(
    os.environ.get("RF_LABELS_PATH", r"E:\\毕业设计\\新测试\\新的\\email_labels.txt")
)
DEFAULT_MODEL_OUTPUT_PATH = Path(
    os.environ.get("RF_MODEL_OUTPUT", r"E:\\毕业设计\\新测试\\随机森林算法模型\\rf_model.joblib")
)


def _ensure_path(path_like: Path | str) -> Path:
    """将输入转换为 Path 并展开用户目录。"""

    return Path(path_like).expanduser()


def validate_paths(features_npy: Path | str, labels_txt: Path | str, model_output: Path | str) -> None:
    """确保特征、标签及模型保存路径有效。"""

    features_path = _ensure_path(features_npy)
    labels_path = _ensure_path(labels_txt)
    model_path = _ensure_path(model_output)

    if not features_path.is_file():
        raise FileNotFoundError(f"未找到特征文件: {features_path}")

    inferred_labels_npy = features_path.with_name(features_path.stem + "_labels.npy")
    if not labels_path.is_file() and not inferred_labels_npy.is_file():
        raise FileNotFoundError(
            f"未找到标签文件: {labels_path}，且未找到配套的 {inferred_labels_npy.name}。"
        )

    if model_path.is_dir():
        raise IsADirectoryError(
            f"模型保存路径必须是文件路径，而不是目录: {model_path}"
        )


def load_features_and_labels(
    features_npy: Path | str, labels_txt: Path | str
) -> tuple[np.ndarray, np.ndarray]:
    """加载特征和标签数据。"""

    features_path = _ensure_path(features_npy)
    labels_path = _ensure_path(labels_txt)

    X = np.load(features_path)

    labels_npy = features_path.with_name(features_path.stem + "_labels.npy")
    if labels_npy.is_file():
        y = np.load(labels_npy)
    else:
        with labels_path.open("r", encoding="utf-8") as fh:
            y = np.array([1 if line.strip().split()[0] == "spam" else 0 for line in fh])

    if X.shape[0] != len(y):
        raise ValueError(
            f"特征与标签数量不一致: X 有 {X.shape[0]} 行，但标签有 {len(y)} 条。"
        )

    unique_labels = np.unique(y)
    if unique_labels.size < 2:
        raise ValueError(f"标签文件中只有一个类别的数据，无法训练模型。唯一标签值: {unique_labels}")

    class_counts = np.bincount(y)
    if class_counts.size < 2 or np.any(class_counts < 2):
        raise ValueError(
            "每个类别至少需要 2 个样本才能进行分层划分。"
            f" 当前各类别样本数: {class_counts.tolist()}"
        )

    return X, y


def train_random_forest_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: str | int | float | None,
    validation_size: float,
    test_size: float,
    random_state: int,
    n_jobs: int,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """训练随机森林分类器并返回模型与验证/测试集预测结果。"""

    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size 与 test_size 必须为正数。")

    temp_size = validation_size + test_size
    if temp_size >= 1:
        raise ValueError(
            "validation_size 与 test_size 之和必须小于 1，"
            f" 当前为 {temp_size:.2f}。"
        )

    if X.shape[0] < 5:
        raise ValueError("样本数量过少，无法按照 70/15/15 划分，请提供更多数据。")

    print("[1/4] 正在划分训练/验证/测试集……")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=random_state,
        stratify=y,
    )

    validation_ratio = validation_size / temp_size
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - validation_ratio,
        random_state=random_state,
        stratify=y_temp,
    )
    print(
        f"完成数据划分：训练 {len(y_train)}，验证 {len(y_valid)}，测试 {len(y_test)}。"
    )

    print("[2/4] 正在初始化模型……")
    model = RandomForestClassifier(
        n_estimators=max(1, n_estimators),
        max_depth=None if max_depth is None or max_depth < 1 else max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    depth_text = "不限制" if model.max_depth is None else str(model.max_depth)
    print(
        "已选择 RandomForestClassifier，"\
        f"n_estimators={model.n_estimators}，max_depth={depth_text}，"\
        f"min_samples_split={min_samples_split}，min_samples_leaf={min_samples_leaf}，"\
        f"max_features={model.max_features}，n_jobs={model.n_jobs}。"
    )

    print("[3/4] 正在训练模型……")
    model.fit(X_train, y_train)
    print("模型训练完成。")

    print("[4/4] 正在评估模型……")
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    return model, y_valid, y_valid_pred, y_test, y_test_pred


def _format_metric_block(
    *,
    dataset_label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: str = "垃圾邮件",
    negative_label: str = "非垃圾邮件",
) -> str:
    """格式化单个数据集的评估指标，贴近示例截图的结构。"""

    report = classification_report(
        y_true,
        y_pred,
        target_names=[negative_label, positive_label],
        output_dict=True,
        zero_division=0,
    )

    accuracy = report["accuracy"]
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1_score = report["weighted avg"]["f1-score"]

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        raise ValueError("混淆矩阵形状异常，期望为 2x2。")

    tn, fp, fn, tp = cm.ravel()

    lines = [
        f"{dataset_label}样本数: {len(y_true)}",
        "",
        "==================== 模型评估结果 ====================",
        "模型评估精度：",
        f"Accuracy:     {accuracy:0.4f} ({accuracy * 100:5.2f}%)",
        f"Precision:    {precision:0.4f} ({precision * 100:5.2f}%)",
        f"Recall:       {recall:0.4f} ({recall * 100:5.2f}%)",
        f"F1-Score:    {f1_score:0.4f} ({f1_score * 100:5.2f}%)",
        "",
        f"预测正确率: {accuracy:0.4f} ({accuracy * 100:5.2f}%)",
        "",
        "混淆矩阵:",
        f"True Positive (预测为{positive_label}, 实际为{positive_label}): {tp}",
        f"True Negative (预测为{negative_label}, 实际为{negative_label}): {tn}",
        f"False Positive (预测为{positive_label}, 实际为{negative_label}): {fp}",
        f"False Negative (预测为{negative_label}, 实际为{positive_label}): {fn}",
    ]

    return "\n".join(lines)


def evaluate_and_report(
    *,
    y_valid: np.ndarray,
    y_valid_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
) -> None:
    """输出验证集和测试集上的评估指标（中文，贴近示例展示风格）。"""

    print("\n==================== 验证集评估 ====================")
    print(
        _format_metric_block(
            dataset_label="验证集",
            y_true=y_valid,
            y_pred=y_valid_pred,
        )
    )

    print("\n==================== 测试集评估 ====================")
    print(
        _format_metric_block(
            dataset_label="测试集",
            y_true=y_test,
            y_pred=y_test_pred,
        )
    )


def save_model(model: RandomForestClassifier, model_output: Path | str) -> None:
    """将训练好的模型保存到指定路径。"""

    output_path = _ensure_path(model_output)
    if output_path.is_dir():
        raise IsADirectoryError(
            f"模型保存路径必须是文件路径，而不是目录: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model}, output_path)
    print(f"模型已保存至: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用随机森林从特征和标签文件训练垃圾邮件分类模型",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--features-npy",
        default=None,
        help=(
            "特征数组 (.npy) 文件路径。"
            "\n- 未提供时将尝试使用默认路径: "
            f"{DEFAULT_FEATURES_PATH}"
            "\n- 也可通过环境变量 RF_FEATURES_PATH 指定默认值"
        ),
    )
    parser.add_argument(
        "--labels-txt",
        default=None,
        help=(
            "标签 (.txt) 文件路径。"
            "\n- 未提供时将尝试使用默认路径: "
            f"{DEFAULT_LABELS_PATH}"
            "\n- 也可通过环境变量 RF_LABELS_PATH 指定默认值"
        ),
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help=(
            "模型保存路径 (.joblib) 文件。"
            "\n- 未提供时将尝试使用默认路径: "
            f"{DEFAULT_MODEL_OUTPUT_PATH}"
            "\n- 也可通过环境变量 RF_MODEL_OUTPUT 指定默认值"
        ),
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="森林中树的数量。",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="树的最大深度，默认不限制。",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="内部节点再划分所需的最小样本数。",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="叶子节点所需的最小样本数。",
    )
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help=(
            "寻找最佳分割时考虑的特征数，可为 int/float/""auto""/""sqrt""/""log2""/None。"
        ),
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.15,
        help="验证集所占比例 (0~1)。默认 0.15，即训练/验证/测试=70/15/15。",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="测试集所占比例 (0~1)。默认 0.15，即训练/验证/测试=70/15/15。",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子，确保结果可复现。",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="并行训练使用的 CPU 数，-1 表示使用所有可用内核。",
    )

    return parser


def resolve_cli_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """合并 CLI 参数与默认设置，缺失时提示所用路径。"""

    def _resolve(value: str | None, default: Path, description: str) -> Path:
        if value:
            return _ensure_path(value)

        resolved_default = _ensure_path(default)
        print(f"未提供{description}，使用默认路径: {resolved_default}")
        return resolved_default

    features_path = _resolve(args.features_npy, DEFAULT_FEATURES_PATH, "特征文件")
    labels_path = _resolve(args.labels_txt, DEFAULT_LABELS_PATH, "标签文件")
    model_output_path = _resolve(args.model_output, DEFAULT_MODEL_OUTPUT_PATH, "模型输出文件")

    return features_path, labels_path, model_output_path


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        features_path, labels_path, model_output_path = resolve_cli_paths(args)
        validate_paths(
            features_npy=features_path,
            labels_txt=labels_path,
            model_output=model_output_path,
        )
        print("开始加载特征与标签……")
        X, y = load_features_and_labels(features_path, labels_path)
        print(
            f"数据加载完成：特征矩阵形状 {X.shape}，标签数量 {y.size}。"
        )

        model, y_valid, y_valid_pred, y_test, y_test_pred = train_random_forest_classifier(
            X,
            y,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            validation_size=args.validation_size,
            test_size=args.test_size,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        evaluate_and_report(
            y_valid=y_valid,
            y_valid_pred=y_valid_pred,
            y_test=y_test,
            y_test_pred=y_test_pred,
        )
        save_model(model, model_output_path)
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        print(f"错误: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
