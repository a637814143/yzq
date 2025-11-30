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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler


# 默认路径，可通过环境变量覆盖
DEFAULT_FEATURES_PATH = Path(
    os.environ.get("NB_FEATURES_PATH", r"E:\\毕业设计\\新测试\新的\\email_features.npy")
)
DEFAULT_LABELS_PATH = Path(
    os.environ.get("NB_LABELS_PATH", r"E:\\毕业设计\\新测试\新的\\email_labels.txt")
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

    inferred_labels_npy = features_path.with_name(features_path.stem + "_labels.npy")
    if not labels_path.is_file() and not inferred_labels_npy.is_file():
        raise FileNotFoundError(
            f"未找到标签文件: {labels_path}，且未找到配套的 {inferred_labels_npy.name}。"
        )

    if model_path.is_dir():
        raise IsADirectoryError(f"模型保存路径必须是文件而不是目录: {model_path}")


def load_dataset(features_npy: Path | str, labels_txt: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """加载特征矩阵 ``X`` 与标签 ``y``。"""

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

    unique = np.unique(y)
    if unique.size < 2:
        raise ValueError(
            "标签文件中仅包含一个类别，无法训练模型。"
            f" 唯一标签值: {unique}"
        )

    class_counts = np.bincount(y)
    if class_counts.size < 2 or np.any(class_counts < 2):
        raise ValueError(
            "每个类别至少需要 2 个样本才能进行分层划分。"
            f" 当前各类别样本数: {class_counts.tolist()}"
        )

    return X, y


def train_naive_bayes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_type: str,
    alpha: float,
    alpha_grid: list[float] | None,
    fit_prior: bool,
    binarize: float | None,
    max_epochs: int,
    patience: int,
    min_delta: float,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> tuple[object, MinMaxScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """训练朴素贝叶斯分类器并返回模型及验证/测试集。"""

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

    print("[1/5] 正在划分训练/验证/测试集……")
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
        "完成数据划分：",
        f"训练集 {len(y_train)} 样本，验证集 {len(y_valid)} 样本，测试集 {len(y_test)} 样本。",
    )

    print("[2/5] 正在缩放特征（MinMaxScaler，确保非负值）……")
    # MultinomialNB 需要非负值，使用 MinMaxScaler 而不是 StandardScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    # 确保没有负值（MinMaxScaler 应该已经保证，但双重检查）
    X_train_scaled = np.maximum(X_train_scaled, 0)
    X_valid_scaled = np.maximum(X_valid_scaled, 0)
    X_test_scaled = np.maximum(X_test_scaled, 0)
    print("特征缩放完成（值范围: 0-1）。")

    def _instantiate(nb_alpha: float):
        if model_type == "multinomial":
            return MultinomialNB(alpha=nb_alpha, fit_prior=fit_prior)
        return BernoulliNB(alpha=nb_alpha, fit_prior=fit_prior, binarize=binarize)

    chosen_alpha = alpha
    if alpha_grid:
        print("[3/5] 正在基于验证集调优 alpha……")
        best_score = -1.0
        for candidate in alpha_grid:
            candidate_model = _instantiate(candidate)
            candidate_model.fit(X_train_scaled, y_train)
            preds = candidate_model.predict(X_valid_scaled)
            score = f1_score(y_valid, preds, average="weighted", zero_division=0)
            print(f"  alpha={candidate:g} -> 验证集加权 F1={score:0.4f}")
            if score > best_score:
                chosen_alpha = candidate
                best_score = score

        print(f"选择验证集表现最佳的 alpha={chosen_alpha:g} (加权 F1={best_score:0.4f})")
    else:
        print("[3/5] 未提供 alpha 网格，直接使用命令行 alpha。")

    print("[4/5] 正在初始化并训练模型……")
    model = _instantiate(chosen_alpha)
    print(f"已选择 {model.__class__.__name__}，alpha={chosen_alpha:g}。")

    unlimited = max_epochs <= 0
    if unlimited:
        print(
            "已取消训练轮次上限，将持续按验证集加权 F1 监测，"
            f"当连续 {patience} 轮未提升超过 min_delta={min_delta} 时停止。"
        )
    else:
        print(
            f"最大训练轮次设为 {max_epochs}，连续 {patience} 轮未提升"
            f"超过 min_delta={min_delta} 时提前停止。"
        )

    classes = np.unique(y_train)
    best_f1 = -np.inf
    best_model_bytes: bytes | None = None
    epochs_without_improve = 0
    epoch = 0
    safety_cap = 5000 if unlimited else max_epochs

    while True:
        epoch += 1
        # BernoulliNB/MultinomialNB 均支持 partial_fit，可近似“多轮训练”并监控指标
        if epoch == 1:
            model.partial_fit(X_train_scaled, y_train, classes=classes)
        else:
            model.partial_fit(X_train_scaled, y_train)

        preds_valid = model.predict(X_valid_scaled)
        f1 = f1_score(y_valid, preds_valid, average="weighted", zero_division=0)
        print(f"  [训练轮次 {epoch}] 验证集加权 F1={f1:0.4f}")

        if f1 > best_f1 + min_delta:
            best_f1 = f1
            epochs_without_improve = 0
            best_model_bytes = joblib.dumps(model)
            print("    指标提升，保存当前模型为最佳。")
        else:
            epochs_without_improve += 1
            print(
                "    指标未显著提升，连续停滞轮数:",
                f"{epochs_without_improve}/{patience}",
            )

        if not unlimited and epoch >= max_epochs:
            print("达到设定的最大训练轮次，停止训练。")
            break
        if epochs_without_improve >= patience:
            print("验证集指标连续未提升，认为已收敛，停止训练。")
            break
        if unlimited and epoch >= safety_cap:
            print(
                "达到安全轮次上限 5000 仍未收敛，为避免无限循环强制停止；"
                "如需继续可显式增大 --max-epochs。"
            )
            break

    if best_model_bytes is not None:
        model = joblib.loads(best_model_bytes)

    print("[5/5] 模型训练完成，并基于验证集选择了最佳轮次。")

    return model, scaler, X_valid_scaled, y_valid, X_test_scaled, y_test


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
    model,
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


def save_model(model, scaler: MinMaxScaler, model_output: Path | str) -> None:
    """保存训练好的模型和标准化器到指定路径。"""

    output_path = _ensure_path(model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存模型和标准化器
    model_data = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(model_data, output_path)
    print(f"模型和标准化器已保存至: {output_path}")


def _parse_binarize(value: str) -> float | None:
    """解析 ``--binarize`` 参数，允许传入 ``none`` 关闭二值化。"""

    if value.lower() == "none":
        return None
    return float(value)


def _parse_alpha_grid(value: str | None) -> list[float] | None:
    """解析 ``--alpha-grid`` 的候选列表。"""

    if not value:
        return None

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        return None

    alphas: list[float] = []
    for part in parts:
        alpha = float(part)
        if alpha < 0:
            raise argparse.ArgumentTypeError("alpha 网格中的值必须为非负数。")
        alphas.append(alpha)

    return alphas


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
        "--alpha-grid",
        type=_parse_alpha_grid,
        default=None,
        help=(
            "逗号分隔的 alpha 候选列表，"
            "如 0.1,0.5,1.0；若提供则会基于验证集选择效果最佳的 alpha。"
        ),
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
        "--max-epochs",
        type=int,
        default=0,
        help=(
            "训练轮次上限；默认 0 表示不设上限，按验证集收敛判断停止。"
            "\n朴素贝叶斯本身无迭代优化，此处使用 partial_fit 循环监控指标，"
            "以满足“训练到精度不再下降”的需求。"
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="验证集加权 F1 连续未提升的容忍轮数，超过则停止训练。",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="将提升视为有效改进的最小增量，避免浮点抖动造成误判。",
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

        model, scaler, X_valid, y_valid, X_test, y_test = train_naive_bayes(
            X,
            y,
            model_type=args.model_type,
            alpha=args.alpha,
            alpha_grid=args.alpha_grid,
            fit_prior=args.fit_prior,
            binarize=args.binarize,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            validation_size=args.validation_size,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        evaluate_and_report(
            model,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
        )
        save_model(model, scaler, model_output_path)
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        print(f"错误: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
