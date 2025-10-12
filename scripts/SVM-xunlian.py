"""使用支持向量机 (SVM) 对邮件特征进行训练的脚本。

该脚本读取 `.npy` 特征文件和 `.txt` 标签文件（spam/ham），
通过线性核的 SVM 训练分类模型，并将训练好的模型保存为 `.joblib` 文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# 默认路径（可根据需要修改）
DEFAULT_FEATURES_PATH = r"E:\毕业设计\新测试\email_features.npy"
DEFAULT_LABELS_PATH = r"E:\毕业设计\trec06c\full\index"
DEFAULT_MODEL_OUTPUT_PATH = r"E:\毕业设计\新测试\支持向量机SVM算法\svm_model.joblib"


def validate_paths(features_npy: str, labels_txt: str, model_output: str) -> None:
    """确保特征、标签及模型保存路径有效。"""

    features_path = Path(features_npy)
    labels_path = Path(labels_txt)
    model_path = Path(model_output)

    if not features_path.is_file():
        raise FileNotFoundError(f"未找到特征文件: {features_path}")

    if not labels_path.is_file():
        raise FileNotFoundError(f"未找到标签文件: {labels_path}")

    if model_path.is_dir():
        raise IsADirectoryError(
            f"模型保存路径必须是文件路径，而不是目录: {model_path}"
        )


def load_features_and_labels(features_npy: str, labels_txt: str) -> tuple[np.ndarray, np.ndarray]:
    """加载特征和标签数据。"""

    features_path = Path(features_npy)
    labels_path = Path(labels_txt)

    X = np.load(features_path)

    with labels_path.open("r", encoding="utf-8") as fh:
        y = np.array([1 if line.strip().split()[0] == "spam" else 0 for line in fh])

    unique_labels = np.unique(y)
    if unique_labels.size < 2:
        raise ValueError(f"标签文件中只有一个类别的数据，无法训练模型。唯一标签值: {unique_labels}")

    return X, y


def train_svm_classifier(X: np.ndarray, y: np.ndarray) -> SVC:
    """使用线性核 SVM 训练分类器。"""

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    print("模型评估：")
    print(classification_report(y_valid, y_pred))
    print("混淆矩阵：")
    print(confusion_matrix(y_valid, y_pred))

    return model


def save_model(model: SVC, model_output: str) -> None:
    """将训练好的模型保存到指定路径。"""

    output_path = Path(model_output)
    if output_path.is_dir():
        raise IsADirectoryError(
            f"模型保存路径必须是文件路径，而不是目录: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"模型已保存至: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 SVM 从特征和标签文件训练垃圾邮件分类模型"
    )

    parser.add_argument(
        "--features-npy",
        default=DEFAULT_FEATURES_PATH,
        help=(
            "特征数组 (.npy) 文件路径，"
            "例如 E:/毕业设计/新测试/email_features.npy"
            f"（默认: {DEFAULT_FEATURES_PATH}）"
        ),
    )
    parser.add_argument(
        "--labels-txt",
        default=DEFAULT_LABELS_PATH,
        help=(
            "标签 (.txt) 文件路径，"
            "例如 E:/毕业设计/trec06c/full/index"
            f"（默认: {DEFAULT_LABELS_PATH}）"
        ),
    )
    parser.add_argument(
        "--model-output",
        default=DEFAULT_MODEL_OUTPUT_PATH,
        help=(
            "模型保存路径 (.joblib) 文件，"
            "例如 E:/毕业项目/scripts/svm_model.joblib"
            f"（默认: {DEFAULT_MODEL_OUTPUT_PATH}）"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        validate_paths(
            features_npy=args.features_npy,
            labels_txt=args.labels_txt,
            model_output=args.model_output,
        )
        X, y = load_features_and_labels(args.features_npy, args.labels_txt)
        model = train_svm_classifier(X, y)
        save_model(model, args.model_output)
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        print(f"错误: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()