"""使用逻辑回归算法训练垃圾邮件分类模型的脚本。

该脚本的使用方式与 ``pusu-xunlian.py`` 保持一致：

* 接收 ``.npy`` 特征文件与 ``.txt`` 标签文件（spam/ham）
* 训练完成后将模型保存为 ``.joblib`` 文件
* 若未显式提供路径，会退回到默认值或环境变量指定的路径

默认采用 :class:`sklearn.linear_model.LogisticRegression`，并暴露常用超参数
（如 ``C``、``penalty``、``max_iter`` 等）供命令行配置。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# 默认路径，可通过环境变量覆盖
DEFAULT_FEATURES_PATH = Path(
    os.environ.get("LR_FEATURES_PATH", r"E:\\毕业设计\\新测试\\新的\\email_features.npy")
)
DEFAULT_LABELS_PATH = Path(
    os.environ.get("LR_LABELS_PATH", r"E:\\毕业设计\\trec06c\\full\\index")
)
DEFAULT_MODEL_OUTPUT_PATH = Path(
    os.environ.get("LR_MODEL_OUTPUT", r"E:\\毕业设计\\新测试\\逻辑回归算法模型\\lr_model.joblib")
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
        raise ValueError("标签文件中仅包含一个类别，无法训练模型。" f" 唯一标签值: {unique}")

    class_counts = np.bincount(y)
    if class_counts.size < 2 or np.any(class_counts < 2):
        raise ValueError(
            "每个类别至少需要 2 个样本才能进行分层划分。"
            f" 当前各类别样本数: {class_counts.tolist()}"
        )

    return X, y


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float,
    penalty: str,
    solver: str,
    max_iter: int,
    validation_size: float,
    test_size: float,
    random_state: int,
    class_weight: str | None,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """训练逻辑回归分类器并返回模型、验证集与测试集。

    默认按照 70% 训练、15% 验证、15% 测试划分；如需调整可通过
    ``validation_size`` 与 ``test_size`` 修改，二者之和需小于 1。
    """

    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size 与 test_size 必须为正数。")

    temp_size = validation_size + test_size
    if temp_size >= 1:
        raise ValueError(
            "validation_size 与 test_size 之和必须小于 1，"
            f" 当前为 {temp_size:.2f}。"
        )

    print("[1/4] 正在划分训练/验证/测试集……")

    if X.shape[0] < 5:
        raise ValueError("样本数量过少，无法按照 70/15/15 划分，请提供更多数据。")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        stratify=y,
        random_state=random_state,
    )

    validation_ratio = validation_size / temp_size
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - validation_ratio,
        stratify=y_temp,
        random_state=random_state,
    )
    print(
        "完成数据划分："
        f"训练集 {len(y_train)} 样本，验证集 {len(y_valid)} 样本，测试集 {len(y_test)} 样本。"
    )

    print("[2/4] 正在初始化模型……")
    _validate_solver_penalty_combination(solver, penalty)

    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
    )
    print(f"已选择 solver={solver}, penalty={penalty}, C={C}。")

    print("[3/4] 正在训练模型……")
    model.fit(X_train, y_train)
    print("模型训练完成。")

    return model, X_valid, y_valid, X_test, y_test


def _validate_solver_penalty_combination(solver: str, penalty: str) -> None:
    """确保 ``solver`` 与 ``penalty`` 组合合法，提前阻断 sklearn 运行时错误。"""

    compatible = {
        "lbfgs": {"l2", "none"},
        "newton-cg": {"l2", "none"},
        "liblinear": {"l1", "l2"},
        "saga": {"l1", "l2", "elasticnet", "none"},
    }

    allowed = compatible.get(solver)
    if allowed is None:
        raise ValueError(f"未知求解器: {solver}")

    if penalty not in allowed:
        readable = ", ".join(sorted(allowed))
        raise ValueError(
            f"求解器 {solver} 与 penalty={penalty} 不兼容。可选: {readable}。"
        )


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
    model: LogisticRegression,
    *,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """输出验证集和测试集上的评估指标（中文，贴近示例展示风格）。"""

    y_valid_pred = model.predict(X_valid)
    print("\n==================== 验证集评估 ====================")
    print(
        _format_metric_block(
            dataset_label="验证集",
            y_true=y_valid,
            y_pred=y_valid_pred,
        )
    )

    y_test_pred = model.predict(X_test)
    print("\n==================== 测试集评估 ====================")
    print(
        _format_metric_block(
            dataset_label="测试集",
            y_true=y_test,
            y_pred=y_test_pred,
        )
    )


def save_model(model: LogisticRegression, model_output: Path | str) -> None:
    """保存训练好的模型到指定路径。"""

    output_path = _ensure_path(model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"模型已保存至: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用逻辑回归训练垃圾邮件分类模型",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--features-npy",
        default=None,
        help=(
            "特征数组 (.npy) 文件路径。"
            "\n- 未提供时将尝试使用默认路径: "
            f"{DEFAULT_FEATURES_PATH}"
            "\n- 也可通过环境变量 LR_FEATURES_PATH 指定默认值"
        ),
    )
    parser.add_argument(
        "--labels-txt",
        default=None,
        help=(
            "标签 (.txt) 文件路径。"
            "\n- 未提供时将尝试使用默认路径: "
            f"{DEFAULT_LABELS_PATH}"
            "\n- 也可通过环境变量 LR_LABELS_PATH 指定默认值"
        ),
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help=(
            "模型保存路径 (.joblib) 文件。"
            "\n- 未提供时将尝试使用默认路径: "
            f"{DEFAULT_MODEL_OUTPUT_PATH}"
            "\n- 也可通过环境变量 LR_MODEL_OUTPUT 指定默认值"
        ),
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="正则化强度的倒数 (C)，更大的值意味着更弱的正则化。",
    )
    parser.add_argument(
        "--penalty",
        choices=("l1", "l2", "elasticnet", "none"),
        default="l2",
        help="正则化类型。请确保与 solver 兼容。",
    )
    parser.add_argument(
        "--solver",
        choices=("lbfgs", "liblinear", "saga", "newton-cg"),
        default="lbfgs",
        help="求解器类型，与 penalty 组合需满足 sklearn 要求。",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="最大迭代次数。",
    )
    parser.add_argument(
        "--class-weight",
        choices=("balanced",),
        default=None,
        help="类别权重设置。传入 balanced 可按频率自动调整。",
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

        model, X_valid, y_valid, X_test, y_test = train_logistic_regression(
            X,
            y,
            C=args.C,
            penalty=args.penalty,
            solver=args.solver,
            max_iter=args.max_iter,
            validation_size=args.validation_size,
            test_size=args.test_size,
            random_state=args.random_state,
            class_weight=args.class_weight,
        )

        evaluate_and_report(
            model,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
        )
        save_model(model, model_output_path)
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        print(f"错误: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
