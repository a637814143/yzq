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
from sklearn.svm import LinearSVC, SVC


# 默认路径（可根据需要修改或通过环境变量覆盖）
DEFAULT_FEATURES_PATH = Path(
    os.environ.get("SVM_FEATURES_PATH", r"E:\\毕业设计\\新测试\\email_features.npy")
)
DEFAULT_LABELS_PATH = Path(
    os.environ.get("SVM_LABELS_PATH", r"E:\\毕业设计\\trec06c\\full\\index")
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

    if not labels_path.is_file():
        raise FileNotFoundError(f"未找到标签文件: {labels_path}")

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

    with labels_path.open("r", encoding="utf-8") as fh:
        y = np.array([1 if line.strip().split()[0] == "spam" else 0 for line in fh])

    unique_labels = np.unique(y)
    if unique_labels.size < 2:
        raise ValueError(f"标签文件中只有一个类别的数据，无法训练模型。唯一标签值: {unique_labels}")

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
    progress_interval: float,
) -> BaseEstimator:
    """使用线性核 SVM 训练分类器。"""

    stage_start = time.perf_counter()
    print("[1/4] 正在划分训练/验证集……")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    split_elapsed = time.perf_counter() - stage_start
    print(f"完成数据划分，用时 {split_elapsed:.2f} 秒。")

    print("[2/4] 正在初始化模型……")
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

    print("[3/4] 正在训练模型……")
    train_start = time.perf_counter()
    with PeriodicStatusPrinter("模型训练", interval=progress_interval):
        model.fit(X_train, y_train)
    train_elapsed = time.perf_counter() - train_start
    print(f"模型训练完成，用时 {train_elapsed:.2f} 秒。")

    print("[4/4] 正在评估模型效果……")
    eval_start = time.perf_counter()
    y_pred = model.predict(X_valid)
    print("模型评估：")
    print(classification_report(y_valid, y_pred))
    print("混淆矩阵：")
    print(confusion_matrix(y_valid, y_pred))
    eval_elapsed = time.perf_counter() - eval_start
    print(f"评估完成，用时 {eval_elapsed:.2f} 秒。")

    if probability and implementation == "svc" and verbose:
        print(
            "提示：启用了概率输出，SVC 需要额外的交叉验证来估计概率，这会显著增加训练时间。"
        )

    return model


def save_model(model: BaseEstimator, model_output: Path | str) -> None:
    """将训练好的模型保存到指定路径。"""

    output_path = _ensure_path(model_output)
    if output_path.is_dir():
        raise IsADirectoryError(
            f"模型保存路径必须是文件路径，而不是目录: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"模型已保存至: {output_path}")


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
        default=1000,
        help=(
            "设定优化的最大迭代次数。"
            "\n- 对 LinearSVC 与 SVC 均生效，设置为 -1 表示不限制。"
        ),
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="优化停止的容忍度，数值越小越精确，但训练可能更慢。",
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
        model = train_svm_classifier(
            X,
            y,
            probability=args.probability,
            verbose=args.verbose,
            implementation=args.implementation,
            max_iter=args.max_iter,
            tol=args.tol,
            progress_interval=args.progress_interval,
        )
        save_model(model, model_output_path)
    except (ValueError, FileNotFoundError, IsADirectoryError) as exc:
        print(f"错误: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()