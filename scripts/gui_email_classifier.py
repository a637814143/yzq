"""简易图形界面：上传邮件文件，选择模型并执行垃圾邮件识别。"""

from __future__ import annotations

import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import joblib
import numpy as np

from email_feature_engine import parser as email_parser
from email_feature_engine import text_features, vectorization

DEFAULT_LR_MODEL = Path(
    os.environ.get("LR_MODEL_OUTPUT", r"E:\\毕业设计\\新测试\\逻辑回归算法模型\\lr_model.joblib")
)
DEFAULT_NB_MODEL = Path(
    os.environ.get("NB_MODEL_OUTPUT", r"E:\\毕业设计\\新测试\\朴素贝叶斯\\nb_model.joblib")
)
DEFAULT_SVM_MODEL = Path(
    os.environ.get("SVM_MODEL_OUTPUT", r"E:\\毕业设计\\新测试\\支持向量机SVM算法\\svm_model.joblib")
)


class EmailClassifierApp(tk.Tk):
    """上传邮件、选择模型并输出预测结果的桌面窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.title("垃圾邮件识别窗口")
        self.geometry("900x650")

        self.selected_email: Path | None = None
        self.model_paths: dict[str, Path] = {
            "逻辑回归模型": DEFAULT_LR_MODEL,
            "朴素贝叶斯模型": DEFAULT_NB_MODEL,
            "SVM 模型": DEFAULT_SVM_MODEL,
            "自定义模型": DEFAULT_LR_MODEL,
        }
        self.loaded_models: dict[Path, object] = {}

        self._build_widgets()

    def _build_widgets(self) -> None:
        padding = {"padx": 10, "pady": 8}

        header = ttk.Label(
            self,
            text="上传邮件（.eml/.txt），选择模型后点击“测试”查看预测与概率",
            font=("微软雅黑", 12, "bold"),
        )
        header.pack(anchor="w", **padding)

        file_frame = ttk.LabelFrame(self, text="1. 邮件上传")
        file_frame.pack(fill="x", **padding)
        ttk.Button(file_frame, text="选择文件", command=self._choose_email).pack(
            side="left", padx=8, pady=6
        )
        self.email_label = ttk.Label(file_frame, text="未选择文件")
        self.email_label.pack(side="left", padx=8)

        model_frame = ttk.LabelFrame(self, text="2. 选择模型")
        model_frame.pack(fill="x", **padding)
        ttk.Label(model_frame, text="模型列表：").pack(side="left", padx=6)

        self.model_var = tk.StringVar(value="逻辑回归模型")
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=list(self.model_paths.keys()),
            state="readonly",
            width=20,
        )
        self.model_combo.pack(side="left", padx=4, pady=6)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        ttk.Button(model_frame, text="浏览模型文件", command=self._choose_model).pack(
            side="left", padx=8
        )
        self.model_path_label = ttk.Label(
            model_frame, text=f"当前: {self.model_paths[self.model_var.get()]}"
        )
        self.model_path_label.pack(side="left", padx=6)

        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", **padding)
        ttk.Button(
            action_frame,
            text="测试",
            command=self._run_inference,
            width=12,
        ).pack(side="left", padx=6)

        self.result_box = tk.Text(self, height=24, wrap="word", font=("等线", 11))
        self.result_box.pack(fill="both", expand=True, **padding)
        self.result_box.insert(
            "1.0",
            "结果将在此显示。请先选择邮件文件和模型。\n",
        )
        self.result_box.config(state="disabled")

    def _choose_email(self) -> None:
        path = filedialog.askopenfilename(
            title="选择邮件文件",
            filetypes=[
                ("Email / 文本", "*.eml *.txt"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        self.selected_email = Path(path)
        self.email_label.config(text=str(self.selected_email))
        self._append_message(f"已选择邮件: {self.selected_email}\n")

    def _choose_model(self) -> None:
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("Joblib / Pickle 模型", "*.joblib *.pkl"), ("所有文件", "*.*")],
        )
        if not path:
            return
        custom_path = Path(path)
        self.model_paths["自定义模型"] = custom_path
        self.model_var.set("自定义模型")
        self.model_combo.set("自定义模型")
        self.model_path_label.config(text=f"当前: {custom_path}")
        self._append_message(f"已切换到自定义模型: {custom_path}\n")

    def _on_model_change(self, _event: tk.Event | None = None) -> None:
        current = self.model_var.get()
        path = self.model_paths.get(current)
        self.model_path_label.config(text=f"当前: {path}")

    def _append_message(self, text: str) -> None:
        self.result_box.config(state="normal")
        self.result_box.insert("end", text)
        self.result_box.see("end")
        self.result_box.config(state="disabled")

    def _load_email_features(self, path: Path) -> tuple[np.ndarray, dict]:
        ext = path.suffix.lower()
        if ext == ".eml":
            parsed = email_parser.parse_eml(str(path))
            body_preview = parsed.get("body", "") or ""
        else:
            body_preview = path.read_text(encoding="utf-8", errors="ignore")
            parsed = {
                "path": str(path),
                "subject": path.stem,
                "from": "",
                "to": [],
                "body": body_preview,
                "raw": None,
                "attachments": 0,
            }
        feat_dict = text_features.extract_text_features(parsed)
        X, _ = vectorization.vectorize_feature_list([feat_dict])
        return X, {"preview": body_preview, "parsed": parsed}

    def _positive_probability(self, model: object, X: np.ndarray) -> float:
        classes = getattr(model, "classes_", None)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            if classes is not None and 1 in classes:
                idx = list(classes).index(1)
                return float(probs[idx])
            return float(np.max(probs))

        if hasattr(model, "decision_function"):
            score = model.decision_function(X)
            raw_score = float(np.array(score).ravel()[0])
            return float(1 / (1 + np.exp(-raw_score)))

        return 0.0

    def _load_model(self, path: Path):
        if path in self.loaded_models:
            return self.loaded_models[path]
        model = joblib.load(path)
        self.loaded_models[path] = model
        return model

    def _run_inference(self) -> None:
        if not self.selected_email:
            messagebox.showwarning("缺少邮件", "请先选择要检测的邮件文件。")
            return

        model_name = self.model_var.get()
        model_path = self.model_paths.get(model_name)
        if not model_path or not model_path.exists():
            messagebox.showerror("模型不存在", f"未找到模型文件: {model_path}")
            return

        try:
            X, meta = self._load_email_features(self.selected_email)
            model = self._load_model(model_path)
            pred = int(model.predict(X)[0])
            proba = self._positive_probability(model, X)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("预测失败", f"处理或预测时出错: {exc}")
            return

        label_text = "垃圾邮件" if pred == 1 else "正常邮件"
        percentage = proba * 100
        parsed = meta["parsed"]
        preview = (meta.get("preview") or "").strip()

        lines = [
            "================ 预测结果 ================",
            f"模型: {model_name} ({model_path})",
            f"邮件文件: {self.selected_email}",
            f"判定: {label_text} | 概率: {percentage:0.2f}%",
            "",
            "================ 邮件内容预览 ================",
            f"主题: {parsed.get('subject', '')}",
            f"发件人: {parsed.get('from', '')}",
            f"收件人数量: {len(parsed.get('to', []))}",
            "正文预览:",
            preview[:800] + ("..." if len(preview) > 800 else ""),
        ]

        self.result_box.config(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", "\n".join(lines) + "\n")
        self.result_box.see("end")
        self.result_box.config(state="disabled")


if __name__ == "__main__":
    app = EmailClassifierApp()
    app.mainloop()