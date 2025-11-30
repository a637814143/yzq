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

DEFAULT_NB_MODEL = Path(
    os.environ.get("NB_MODEL_OUTPUT", r"E:\\毕业设计\\新测试\\朴素贝叶斯\\nb_model.joblib")
)
DEFAULT_SVM_MODEL = Path(
    os.environ.get("SVM_MODEL_OUTPUT", r"E:\\毕业设计\\新测试\\支持向量机SVM算法\\svm_model.joblib")
)

# 双模型融合的权重（朴素贝叶斯 60%，SVM 40%）
MODEL_WEIGHTS: dict[str, float] = {
    "朴素贝叶斯模型": 0.6,
    "SVM 模型": 0.4,
}


class EmailClassifierApp(tk.Tk):
    """上传邮件、选择模型并输出预测结果的桌面窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.title("垃圾邮件识别窗口")
        self.geometry("900x650")

        self.selected_email: Path | None = None
        self.model_paths: dict[str, Path] = {
            "朴素贝叶斯模型": DEFAULT_NB_MODEL,
            "SVM 模型": DEFAULT_SVM_MODEL,
        }
        self.loaded_models: dict[Path, object] = {}

        self._build_widgets()

    def _build_widgets(self) -> None:
        padding = {"padx": 10, "pady": 8}

        header = ttk.Label(
            self,
            text=(
                "上传邮件（.eml/.txt/.text 等文本）或直接粘贴邮件正文，"
                "支持双模型加权融合：朴素贝叶斯 60%，SVM 40%"
            ),
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

        text_frame = ttk.LabelFrame(self, text="2. 粘贴邮件正文（可选）")
        text_frame.pack(fill="both", expand=False, **padding)
        ttk.Label(text_frame, text="可直接粘贴纯文本邮件内容，若同时提供文件则优先使用文件。 ").pack(
            anchor="w", padx=6, pady=2
        )
        self.manual_text = tk.Text(text_frame, height=6, wrap="word", font=("等线", 11))
        self.manual_text.pack(fill="x", padx=6, pady=4)

        model_frame = ttk.LabelFrame(self, text="3. 模型路径（默认双模型加权融合）")
        model_frame.pack(fill="x", **padding)
        ttk.Label(
            model_frame,
            text="将按 60% 朴素贝叶斯 + 40% SVM 进行融合。",
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(4, 6))

        self.model_labels: dict[str, ttk.Label] = {}
        for idx, (name, key) in enumerate(
            [
                ("朴素贝叶斯", "朴素贝叶斯模型"),
                ("SVM", "SVM 模型"),
            ]
        ):
            ttk.Label(model_frame, text=f"{name} 模型路径:").grid(
                row=idx + 1, column=0, sticky="w", padx=6, pady=4
            )
            label = ttk.Label(model_frame, text=str(self.model_paths[key]))
            label.grid(row=idx + 1, column=1, sticky="w", padx=6, pady=4)
            self.model_labels[key] = label
            ttk.Button(
                model_frame,
                text="浏览",
                command=lambda k=key: self._choose_model(k),
            ).grid(row=idx + 1, column=2, sticky="w", padx=6, pady=4)

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
            "结果将在此显示。请先上传邮件文件或粘贴文本（文本文件可直接选择），然后点击测试执行双模型融合预测。\n",
        )
        self.result_box.config(state="disabled")

    def _choose_email(self) -> None:
        path = filedialog.askopenfilename(
            title="选择邮件文件",
            filetypes=[
                ("Email / 文本", "*.eml *.txt *.text *.log *.md"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        self.selected_email = Path(path)
        self.email_label.config(text=str(self.selected_email))
        self._append_message(f"已选择邮件: {self.selected_email}\n")

    def _choose_model(self, model_key: str) -> None:
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("Joblib / Pickle 模型", "*.joblib *.pkl"), ("所有文件", "*.*")],
        )
        if not path:
            return
        custom_path = Path(path)
        self.model_paths[model_key] = custom_path
        self.model_labels[model_key].config(text=str(custom_path))
        self._append_message(f"已更新 {model_key} 路径: {custom_path}\n")

    def _append_message(self, text: str) -> None:
        self.result_box.config(state="normal")
        self.result_box.insert("end", text)
        self.result_box.see("end")
        self.result_box.config(state="disabled")

    def _load_email_features(self, path: Path | None, manual_text: str) -> tuple[np.ndarray, dict]:
        if path:
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
        else:
            body_preview = manual_text
            parsed = {
                "path": "粘贴文本",
                "subject": "",
                "from": "",
                "to": [],
                "body": manual_text,
                "raw": None,
                "attachments": 0,
            }
        feat_dict = text_features.extract_text_features(parsed)
        X, _, _ = vectorization.vectorize_feature_list([feat_dict])
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
        loaded = joblib.load(path)
        # 检查是否是包含 model/scaler/metadata 的新格式
        if isinstance(loaded, dict) and "model" in loaded and "scaler" in loaded:
            model_data = {
                "model": loaded["model"],
                "scaler": loaded.get("scaler"),
                "metadata": loaded.get("metadata", {}),
            }
        else:
            # 旧格式：直接是模型，scaler 为 None；缺少元数据时推理前会提示
            model_data = {"model": loaded, "scaler": None, "metadata": {}}
        self.loaded_models[path] = model_data
        return model_data

    def _run_inference(self) -> None:
        manual_text = self.manual_text.get("1.0", "end").strip()
        if not self.selected_email and not manual_text:
            messagebox.showwarning("缺少邮件", "请上传邮件文件或粘贴邮件正文后再测试。")
            return

        email_path = self.selected_email
        if email_path and not email_path.exists():
            if manual_text:
                self._append_message("选择的邮件文件不存在，已改用粘贴文本进行检测。\n")
                email_path = None
            else:
                messagebox.showerror("邮件不存在", f"未找到邮件文件: {email_path}")
                return

        missing = [name for name in MODEL_WEIGHTS if not self.model_paths[name].exists()]
        if missing:
            messagebox.showerror(
                "模型不存在",
                "未找到以下模型文件：\n" + "\n".join(f"- {m}: {self.model_paths[m]}" for m in missing),
            )
            return

        try:
            X, meta = self._load_email_features(email_path, manual_text)
            results = []
            fused_proba = 0.0
            total_weight = 0.0
            for model_key, weight in MODEL_WEIGHTS.items():
                model_path = self.model_paths[model_key]
                model_data = self._load_model(model_path)
                model = model_data["model"]
                scaler = model_data["scaler"]
                meta = model_data.get("metadata", {})
                expected_dim = meta.get("n_features")
                if expected_dim is not None and X.shape[1] != expected_dim:
                    raise ValueError(
                        f"模型 {model_key} 期望特征维度为 {expected_dim}，"
                        f"但当前提取得到 {X.shape[1]}。请确认训练和推理使用的特征提取配置一致。"
                    )
                if scaler is None:
                    self._append_message(
                        (
                            f"⚠️ 模型 {model_key} 缺少标准化器，预测可能与训练不一致，"
                            "建议重新训练生成包含 scaler 的模型文件。\n"
                        )
                    )
                X_scaled = scaler.transform(X) if scaler is not None else X
                proba = self._positive_probability(model, X_scaled)
                pred = int(model.predict(X_scaled)[0])
                fused_proba += weight * proba
                total_weight += weight
                results.append((model_key, model_path, pred, proba, weight))
            proba = fused_proba / total_weight if total_weight else 0.0
            pred = int(proba >= 0.5)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("预测失败", f"处理或预测时出错: {exc}")
            return

        label_text = "垃圾邮件" if pred == 1 else "正常邮件"
        percentage = proba * 100
        parsed = meta["parsed"]
        preview = (meta.get("preview") or "").strip()

        source_desc = str(email_path) if email_path else "粘贴文本（未选择文件）"

        lines = [
            "================ 预测结果（加权融合） ================",
            "融合权重: 朴素贝叶斯 60% | SVM 40%",
            f"输入来源: {source_desc}",
            f"判定: {label_text} | 概率: {percentage:0.2f}%",
            "",
            "各模型输出：",
        ]

        for model_key, model_path, m_pred, m_proba, weight in results:
            model_label = "垃圾邮件" if m_pred == 1 else "正常邮件"
            lines.append(
                f"- {model_key} ({model_path}) | 权重 {weight:.2f} | 判定: {model_label} | 概率: {m_proba*100:0.2f}%"
            )

        lines.extend(
            [
            "",
            "================ 邮件内容预览 ================",
            f"主题: {parsed.get('subject', '')}",
            f"发件人: {parsed.get('from', '')}",
            f"收件人数量: {len(parsed.get('to', []))}",
            "正文预览:",
            preview[:800] + ("..." if len(preview) > 800 else ""),
            ]
        )

        self.result_box.config(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", "\n".join(lines) + "\n")
        self.result_box.see("end")
        self.result_box.config(state="disabled")


if __name__ == "__main__":
    app = EmailClassifierApp()
    app.mainloop()