"""使用朴素贝叶斯算法训练垃圾邮件分类模型的脚本。

该脚本与 ``SVM-xunlian.py`` 的使用方式保持一致：

* 接收 ``.npy`` 特征文件与 ``.txt`` 标签文件（spam/ham）
* 训练完成后将模型保存为 ``.joblib`` 文件
* 若未显式提供路径，会退回到默认值或环境变量指定的路径

默认采用 :class:`sklearn.naive_bayes.MultinomialNB`，并提供若干常用超参数
（如 ``alpha``、``fit_prior`` 等）供命令行配置。
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
from sklearn.naive_bayes import BernoulliNB, MultinomialNB


# 默认路径，可通过环境变量覆盖
DEFAULT_FEATURES_PATH = Path(
    os.environ.get("NB_FEATURES_PATH", r"E:\\毕业设计\\新测试\\email_features.npy")
)
DEFAULT_LABELS_PATH = Path(
    os.environ.get("NB_LABELS_PATH", r"E:\\毕业设计\\trec06c\\full\\index")
)
DEFAULT_MODEL_OUTPUT_PATH = Path(
    os.environ.get("NB_MODEL_OUTPUT", r"E:\\毕业设计\\新测试\朴素贝叶斯\\nb_model.joblib")
)


def _ensure_path(path_like: Path | str) -> Path:
    """将输入转换为 ``Path`` 并展开用户目录。"""

    return Path(path_like).expanduser()


def validate_paths(features_npy: Path | str, labels_txt: Path | str, model_output: Path | str) -> None:
    """确保特征、标签及模型输出路径有效。"""

    features_path = _ensure_path(features_npy)
    labels_path = _ensure_path(labels_txt)
    model_path = _ensure_path(model_output)

    if not features_path.is_file():
        raise FileNotFoundError(f"未找到特征文件: {features_path}")

    if not labels_path.is_file():
        raise FileNotFoundError(f"未找到标签文件: {labels_path}")

    if model_path.is_dir():
        raise IsADirectoryError(f"模型保存路径必须是文件而不是目录: {model_path}")


def load_dataset(features_npy: Path | str, labels_txt: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """加载特征矩阵 ``X`` 与标签 ``y``。"""

    features_path = _ensure_path(features_npy)
    labels_path = _ensure_path(labels_txt)

    X = np.load(features_path)

    with labels_path.open("r", encoding="utf-8") as fh:
        y = np.array([1 if line.strip().split()[0] == "spam" else 0 for line in fh])

    unique = np.unique(y)
    if unique.size < 2:
        raise ValueError(
            "标签文件中仅包含一个类别，无法训练模型。"
            f" 唯一标签值: {unique}"
        )

    return X, y


def train_naive_bayes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: str,
    alpha: float,
    fit_prior: bool,
    binarize: float | None,
    test_size: float,
    random_state: int,
) -> tuple[object, np.ndarray, np.ndarray]:
    """训练朴素贝叶斯分类器并返回模型及验证集。"""

    print("[1/3] 正在划分训练/验证集……")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print("完成数据划分。")

    print("[2/3] 正在初始化模型……")
    if model_type == "multinomial":
        model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    else:
        model = BernoulliNB(alpha=alpha, fit_prior=fit_prior, binarize=binarize)
    print(f"已选择 {model.__class__.__name__}。")

    print("[3/3] 正在训练模型……")
    model.fit(X_train, y_train)
    print("模型训练完成。")

    return model, X_valid, y_valid


def evaluate_and_report(model, X_valid: np.ndarray, y_valid: np.ndarray) -> None:
    """输出验证集上的评估指标。"""

    y_pred = model.predict(X_valid)
    print("模型评估：")
    print(classification_report(y_valid, y_pred))
    print("混淆矩阵：")
    print(confusion_matrix(y_valid, y_pred))


def save_model(model, model_output: Path | str) -> None:
    """保存训练好的模型到指定路径。"""

    output_path = _ensure_path(model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"模型已保存至: {output_path}")


def _parse_binarize(value: str) -> float | None:
    """解析 ``--binarize`` 参数，允许传入 ``none`` 关闭二值化。"""

    if value.lower() == "none":
        return None
    return float(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用朴素贝叶斯训练垃圾邮件分类模型",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--features-npy",
        default=None,
        help=(
            "特征数组 (.npy) 文件路径。"
            "\n- 未提供时将尝试使用默认路径: "
            f"{DEFAULT_FEATURES_PATH}"
            "\n- 也可通过环境变量 NB_FEATURES_PATH 指定默认值"
        ),
    )
    parser.add_argument(
        "--labels-txt",
        default=None,
        help=(
            "标签 (.txt) 文件路径。"
            "\n- 未提供时将尝试使用默认路径: "
            f"{DEFAULT_LABELS_PATH}"
            "\n- 也可通过环境变量 NB_LABELS_PATH 指定默认值"
        ),
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help=(
            "模型保存路径 (.joblib) 文件。"
            "\n- 未提供时将尝试使用默认路径: "
            f"{DEFAULT_MODEL_OUTPUT_PATH}"
            "\n- 也可通过环境变量 NB_MODEL_OUTPUT 指定默认值"
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=("multinomial", "bernoulli"),
        default="multinomial",
        help="选择朴素贝叶斯模型类型：multinomial 或 bernoulli。",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="拉普拉斯/李德斯通平滑参数 (alpha)。",
    )
    parser.add_argument(
        "--fit-prior",
        action="store_true",
        help="根据数据估计先验概率 (默认行为)。",
    )
    parser.add_argument(
        "--no-fit-prior",
        dest="fit_prior",
        action="store_false",
        help="不估计先验概率，使用统一先验。",
    )
    parser.set_defaults(fit_prior=True)
    parser.add_argument(
        "--binarize",
        type=_parse_binarize,
        default=_parse_binarize("0.0"),
        help=(
            "仅对 BernoulliNB 生效，将特征按该阈值二值化。"
            "\n- 传入 none 表示不进行二值化。"
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="验证集所占比例 (0~1)。",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子，确保结果可复现。",
    )

    return parser


def resolve_cli_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """若未显式传入路径，则回退到默认值并提示。"""

    def _resolve(value: str | None, default: Path, description: str) -> Path:
        if value:
            return _ensure_path(value)

        resolved = _ensure_path(default)
        print(f"未提供{description}，使用默认路径: {resolved}")
        return resolved

    features_path = _resolve(args.features_npy, DEFAULT_FEATURES_PATH, "特征文件")
    labels_path = _resolve(args.labels_txt, DEFAULT_LABELS_PATH, "标签文件")
    model_output_path = _resolve(args.model_output, DEFAULT_MODEL_OUTPUT_PATH, "模型输出文件")

    return features_path, labels_path, model_output_path


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        features_path, labels_path, model_output_path = resolve_cli_paths(args)
        validate_paths(features_path, labels_path, model_output_path)

        print("开始加载特征与标签……")
        X, y = load_dataset(features_path, labels_path)
        print(f"数据加载完成：特征矩阵形状 {X.shape}，标签数量 {y.size}。")

        model, X_valid, y_valid = train_naive_bayes(
            X,
            y,
            model_type=args.model_type,
            alpha=args.alpha,
            fit_prior=args.fit_prior,
            binarize=args.binarize,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        evaluate_and_report(model, X_valid, y_valid)
        save_model(model, model_output_path)
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        print(f"错误: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()