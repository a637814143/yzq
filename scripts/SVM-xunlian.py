"""使用支持向量机 (SVM) 对邮件特征进行训练的脚本。

该脚本读取 `.npy` 特征文件和 `.txt` 标签文件（spam/ham），
默认使用 LinearSVC（线性核支持向量机）进行训练，并将训练好的模型保存为 `.joblib` 文件。
如需使用 sklearn 的 SVC 实现或启用概率输出，可通过命令行参数进行配置。
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


# 默认路径（可根据需要修改或通过环境变量覆盖）
DEFAULT_FEATURES_PATH = Path(
    os.environ.get("SVM_FEATURES_PATH", r"E:\\毕业设计\\新测试\\新的\\email_features.npy")
)
DEFAULT_LABELS_PATH = Path(
    os.environ.get("SVM_LABELS_PATH", r"E:\\毕业设计\\新测试\\新的\\email_labels.txt")
)
DEFAULT_MODEL_OUTPUT_PATH = Path(
    os.environ.get("SVM_MODEL_OUTPUT", r"E:\\毕业设计\\新测试\\支持向量机SVM算法\\svm_model.joblib")
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


class PeriodicStatusPrinter:
    """周期性打印提示信息，防止长时间无输出时被误认为卡住。"""

    def __init__(self, label: str, interval: float = 5.0) -> None:
        self.label = label
        self.interval = max(0.5, float(interval))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start_time: float | None = None

    def _run(self) -> None:
        counter = 1
        while not self._stop_event.wait(self.interval):
            if self._start_time is None:
                continue
            elapsed = time.perf_counter() - self._start_time
            print(
                f"... {self.label}仍在进行中 (累计耗时 {elapsed:.1f} 秒，第 {counter} 次提示)"
            )
            sys.stdout.flush()
            counter += 1

    def __enter__(self) -> "PeriodicStatusPrinter":
        self._start_time = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self._stop_event.set()
        self._thread.join()


def train_svm_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    probability: bool,
    verbose: bool,
    implementation: str,
    max_iter: int,
    tol: float,
    validation_size: float,
    test_size: float,
    random_state: int,
    progress_interval: float,
) -> BaseEstimator:
    """使用线性核 SVM 训练分类器。"""

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

    stage_start = time.perf_counter()
    print("[1/6] 正在划分训练/验证/测试集……")
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
    split_elapsed = time.perf_counter() - stage_start
    print(
        f"完成数据划分，用时 {split_elapsed:.2f} 秒。训练 {len(y_train)}，验证 {len(y_valid)}，测试 {len(y_test)}。"
    )

    print("[2/6] 正在标准化特征……")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    print("特征标准化完成。")

    print("[3/6] 正在初始化模型……")
    init_start = time.perf_counter()
    n_samples, n_features = X_train.shape

    if implementation == "linear":
        dual = n_samples < n_features
        base_model = LinearSVC(
            C=1.0,
            max_iter=max_iter,
            tol=tol,
            dual=dual,
            random_state=42,
            verbose=1 if verbose else 0,
        )
        model: BaseEstimator
        if probability:
            if verbose:
                print(
                    "LinearSVC 不原生支持概率，将使用 Platt 缩放进行校准，耗时会增加。"
                )
            model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
        else:
            model = base_model
    else:
        model = SVC(
            kernel="linear",
            C=1.0,
            probability=probability,
            random_state=42,
            tol=tol,
            max_iter=max_iter,
            verbose=1 if verbose else False,
        )

    init_elapsed = time.perf_counter() - init_start
    print(f"模型初始化完成，用时 {init_elapsed:.2f} 秒。")

    print("[4/6] 正在训练模型……")
    if max_iter == -1:
        print(
            "已取消迭代次数上限，优化将根据收敛容忍度自动停止，确保充分训练。"
        )
    train_start = time.perf_counter()
    with PeriodicStatusPrinter("模型训练", interval=progress_interval):
        model.fit(X_train_scaled, y_train)
    train_elapsed = time.perf_counter() - train_start
    print(f"模型训练完成，用时 {train_elapsed:.2f} 秒。")
    n_iter_info = getattr(model, "n_iter_", None)
    if n_iter_info is None and isinstance(model, CalibratedClassifierCV):
        n_iter_info = getattr(model.base_estimator_, "n_iter_", None)
    if n_iter_info is not None:
        iter_values = np.atleast_1d(n_iter_info)
        iter_text = ", ".join(str(int(value)) for value in iter_values)
        print(f"实际迭代轮次: {iter_text}")
    else:
        print(
            "本次训练的底层估计器未暴露 n_iter_ 属性，迭代次数由 max_iter/收敛容忍度决定。"
        )

    print("[5/6] 正在评估模型效果（验证集）……")
    eval_start = time.perf_counter()
    y_valid_pred = model.predict(X_valid_scaled)
    valid_elapsed = time.perf_counter() - eval_start
    print(f"验证集评估完成，用时 {valid_elapsed:.2f} 秒。")

    print("[6/6] 正在评估模型效果（测试集）……")
    test_start = time.perf_counter()
    y_test_pred = model.predict(X_test_scaled)
    test_elapsed = time.perf_counter() - test_start
    print(f"测试集评估完成，用时 {test_elapsed:.2f} 秒。")

    if probability and implementation == "svc" and verbose:
        print(
            "提示：启用了概率输出，SVC 需要额外的交叉验证来估计概率，这会显著增加训练时间。"
        )

    return model, scaler, y_valid, y_valid_pred, y_test, y_test_pred


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


def save_model(model: BaseEstimator, scaler: StandardScaler, model_output: Path | str) -> None:
    """将训练好的模型和标准化器保存到指定路径。"""

    output_path = _ensure_path(model_output)
    if output_path.is_dir():
        raise IsADirectoryError(
            f"模型保存路径必须是文件路径，而不是目录: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存模型和标准化器
    model_data = {
        'model': model,
        'scaler': scaler
    }
    joblib.dump(model_data, output_path)
    print(f"模型和标准化器已保存至: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 SVM 从特征和标签文件训练垃圾邮件分类模型",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--features-npy",
        default=None,
        help=(
            "特征数组 (.npy) 文件路径。"
            "\n- 未提供时将尝试使用默认路径: "
            f"{DEFAULT_FEATURES_PATH}"
            "\n- 也可通过环境变量 SVM_FEATURES_PATH 指定默认值"
        ),
    )
    parser.add_argument(
        "--labels-txt",
        default=None,
        help=(
            "标签 (.txt) 文件路径。"
            "\n- 未提供时将尝试使用默认路径: "
            f"{DEFAULT_LABELS_PATH}"
            "\n- 也可通过环境变量 SVM_LABELS_PATH 指定默认值"
        ),
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help=(
            "模型保存路径 (.joblib) 文件。"
            "\n- 未提供时将尝试使用默认路径: "
            f"{DEFAULT_MODEL_OUTPUT_PATH}"
            "\n- 也可通过环境变量 SVM_MODEL_OUTPUT 指定默认值"
        ),
    )
    parser.add_argument(
        "--probability",
        dest="probability",
        action="store_true",
        help=(
            "启用概率输出 (model.predict_proba)，训练时需要额外计算，可能显著变慢。"
        ),
    )
    parser.add_argument(
        "--no-probability",
        dest="probability",
        action="store_false",
        help="禁用概率输出（默认），训练速度更快。",
    )
    parser.set_defaults(probability=False)
    parser.add_argument(
        "--implementation",
        choices=("linear", "svc"),
        default="linear",
        help=(
            "选择底层 SVM 实现：linear 使用 LinearSVC（更快，默认），"
            "svc 使用核方法 SVC（支持概率且更灵活）。"
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=-1,
        help=(
            "设定优化的最大迭代次数。"
            "\n- 对 LinearSVC 与 SVC 均生效，默认 -1 表示不限制，由收敛容忍度自动停止。"
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="优化停止的容忍度，数值越小越精确，但训练可能更慢。",
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
        "--verbose",
        action="store_true",
        help=(
            "输出更详细的训练过程信息。"
            "\n- 对 LinearSVC 会打印 liblinear 的迭代日志。"
            "\n- 对 SVC 则显示 SMO 优化的进度。"
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=8.0,
        help=(
            "训练阶段的进度提示间隔（秒）。"
            "\n- 当模型训练耗时较长时，会周期性打印提示以确认程序正在运行。"
        ),
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
        if args.implementation == "linear" and args.probability:
            print(
                "注意：LinearSVC 需要额外的概率校准过程，训练时间可能变长。"
            )
        model, scaler, y_valid, y_valid_pred, y_test, y_test_pred = train_svm_classifier(
            X,
            y,
            probability=args.probability,
            verbose=args.verbose,
            implementation=args.implementation,
            max_iter=args.max_iter,
            tol=args.tol,
            validation_size=args.validation_size,
            test_size=args.test_size,
            random_state=args.random_state,
            progress_interval=args.progress_interval,
        )
        evaluate_and_report(
            y_valid=y_valid,
            y_valid_pred=y_valid_pred,
            y_test=y_test,
            y_test_pred=y_test_pred,
        )
        save_model(model, scaler, model_output_path)
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        print(f"错误: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
