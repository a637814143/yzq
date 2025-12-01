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
DEFAULT_LR_MODEL = Path(
    os.environ.get("LR_MODEL_OUTPUT", r"E:\\毕业设计\\新测试\\逻辑回归算法模型\\lr_model.joblib")
)
DEFAULT_DT_MODEL = Path(
    os.environ.get("DT_MODEL_OUTPUT", r"E:\\毕业设计\\新测试\\决策树算法模型\\dt_model.joblib")
)
DEFAULT_RF_MODEL = Path(
    os.environ.get("RF_MODEL_OUTPUT", r"E:\\毕业设计\\新测试\\随机森林算法模型\\rf_model.joblib")
)

# 五模型加权占比（总和为 1.0）
MODEL_WEIGHTS: dict[str, float] = {
    "随机森林模型": 0.50,
    "决策树模型": 0.10,
    "逻辑回归模型": 0.10,
    "SVM 模型": 0.10,
    "朴素贝叶斯模型": 0.20,
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
            "逻辑回归模型": DEFAULT_LR_MODEL,
            "决策树模型": DEFAULT_DT_MODEL,
            "随机森林模型": DEFAULT_RF_MODEL,
        }
        self.loaded_models: dict[Path, object] = {}

        self._build_widgets()

    def _build_widgets(self) -> None:
        padding = {"padx": 10, "pady": 8}

        header = ttk.Label(
            self,
            text="上传邮件或粘贴正文，使用朴素贝叶斯 / SVM / 逻辑回归 / 决策树 / 随机森林 五模型加权识别",
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

        model_frame = ttk.LabelFrame(self, text="3. 确认模型路径与占比")
        model_frame.pack(fill="x", **padding)
        ttk.Label(
            model_frame,
            text=(
                "将按以下占比对五个模型的垃圾邮件概率加权，得到最终判定：\n"
                "随机森林 50% | 决策树 10% | 逻辑回归 10% | SVM 10% | 朴素贝叶斯 20%"
            ),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(4, 6))

        self.model_labels: dict[str, ttk.Label] = {}
        for idx, key in enumerate(self.model_paths.keys(), start=1):
            ttk.Label(model_frame, text=f"{key} 路径:").grid(
                row=idx, column=0, sticky="w", padx=6, pady=4
            )
            label = ttk.Label(model_frame, text=str(self.model_paths[key]))
            label.grid(row=idx, column=1, sticky="w", padx=6, pady=4)
            self.model_labels[key] = label
            ttk.Button(
                model_frame,
                text="浏览",
                command=lambda k=key: self._choose_model(k),
            ).grid(row=idx, column=2, sticky="w", padx=6, pady=4)

        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", **padding)
        ttk.Button(
            action_frame,
            text="开始识别",
            command=self._run_inference,
            width=12,
        ).pack(side="left", padx=6)

        self.result_box = tk.Text(self, height=24, wrap="word", font=("等线", 11))
        self.result_box.pack(fill="both", expand=True, **padding)
        self.result_box.insert(
            "1.0",
            "结果将在此显示。请先上传邮件文件或粘贴文本（文本文件可直接选择），然后点击“开始识别”执行五模型加权判断。\n",
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

    def _load_email_features(
        self, path: Path | None, manual_text: str
    ) -> tuple[np.ndarray, dict]:
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
        return X, {"preview": body_preview, "parsed": parsed, "features": feat_dict}

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

        # 支持以下三种格式：
        # 1) {'model': clf, 'scaler': scaler}（带标准化器）
        # 2) {'model': clf}（仅模型，决策树/随机森林脚本的保存格式）
        # 3) 直接保存的模型对象
        if isinstance(loaded, dict):
            model = loaded.get("model", loaded)
            scaler = loaded.get("scaler")
        else:
            model = loaded
            scaler = None

        model_data = {"model": model, "scaler": scaler}
        self.loaded_models[path] = model_data
        return model_data

    def _probability_method(self, model: object) -> str:
        """给出当前模型输出“垃圾”概率的依据说明。"""
        if hasattr(model, "predict_proba"):
            classes = getattr(model, "classes_", None)
            if classes is not None and 1 in classes:
                return "概率来源: predict_proba，直接取标签 1 对应的概率"
            return "概率来源: predict_proba，未显式包含标签 1 时取最大概率的类别"  # 兼容未显式包含 1 的场景

        if hasattr(model, "decision_function"):
            return "概率来源: decision_function 得分经 sigmoid 转换"

        return "概率来源: 模型未提供概率接口，返回 0 兜底"

    def _feature_basis(self, feat: dict) -> str:
        """根据提取的文本特征给出简要判定依据。"""

        risk_kw_total = int(feat.get("risk_kw_total", 0))
        url_count = int(feat.get("url_count", 0))
        shortlink_count = int(feat.get("shortlink_count", 0))
        money_symbol_count = int(feat.get("money_symbol_count", 0))
        subj_risk = "是" if feat.get("subject_risk_kw") else "否"
        top_unigrams = feat.get("subj_top_unigrams") or []
        top_unigram_desc = (
            "；高频词: "
            + "、".join(f"{tok}({cnt})" for tok, cnt in top_unigrams)
            if top_unigrams
            else "；高频词: 无明显高频词"
        )

        parts = [
            f"风险关键词出现 {risk_kw_total} 次",  # 风险词越多通常越偏向垃圾
            f"URL 数量 {url_count} (短链 {shortlink_count})",  # 带链接/短链可能与垃圾相关
            f"金额符号 {money_symbol_count} 个",  # 金额符号较多可能暗示诈骗或推销
            f"主题含风险词: {subj_risk}",
        ]

        return "依据: " + "；".join(parts) + top_unigram_desc

    def _predict_single_model(
        self, model_key: str, model_path: Path, X: np.ndarray
    ) -> tuple[int, float, str]:
        model_data = self._load_model(model_path)
        model = model_data["model"]
        scaler = model_data["scaler"]
        X_scaled = scaler.transform(X) if scaler is not None else X
        proba_method = self._probability_method(model)
        proba = self._positive_probability(model, X_scaled)
        pred = int(model.predict(X_scaled)[0])
        return pred, proba, proba_method

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

        try:
            X, meta = self._load_email_features(email_path, manual_text)
            feature_basis = self._feature_basis(meta["features"])
            missing_models = [
                key for key, path in self.model_paths.items() if not path.exists()
            ]
            if missing_models:
                messagebox.showerror(
                    "模型不存在",
                    "以下模型文件未找到，请确认路径后重试:\n" + "\n".join(missing_models),
                )
                return

            per_model_results: list[str] = []
            weighted_sum = 0.0
            total_weight = sum(MODEL_WEIGHTS.values()) or 1.0

            # 逐个模型生成“垃圾”概率，并按设定权重求加权平均：
            # 加权概率 = sum(模型垃圾概率 * 模型权重) / 权重总和。
            for model_key, weight in MODEL_WEIGHTS.items():
                path = self.model_paths[model_key]
                pred, proba, proba_method = self._predict_single_model(
                    model_key, path, X
                )
                weighted_sum += proba * weight
                per_model_results.append(
                    " | ".join(
                        [
                            f"{model_key} -> 判定: {'垃圾' if pred == 1 else '正常'}",
                            f"概率: {proba*100:0.2f}%",
                            f"权重: {weight*100:0.0f}%",
                            proba_method,
                            feature_basis,
                            f"路径: {path}",
                        ]
                    )
                )

            agg_proba = weighted_sum / total_weight
            pred = 1 if agg_proba >= 0.5 else 0
            proba = agg_proba
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("预测失败", f"处理或预测时出错: {exc}")
            return

        label_text = "垃圾邮件" if pred == 1 else "正常邮件"
        percentage = proba * 100
        parsed = meta["parsed"]
        preview = (meta.get("preview") or "").strip()

        source_desc = str(email_path) if email_path else "粘贴文本（未选择文件）"

        lines = [
            "================ 预测结果（五模型加权） ================",
            "占比: 随机森林50% + 决策树10% + 逻辑回归10% + SVM 10% + 朴素贝叶斯20%",
            f"输入来源: {source_desc}",
            f"判定: {label_text} | 概率: {percentage:0.2f}%",
            "",
        ]

        lines.extend([
            "================ 各模型判定详情 ================",
            *per_model_results,
            "",
            "================ 概率计算说明 ================",
            "1) 单模型概率：优先用 predict_proba 读取“标签为 1(垃圾)”的列；若模型无标签 1 列，则取概率最大的类别。",
            "2) 若模型不支持 predict_proba 但有 decision_function，则取分值并经过 sigmoid 变换得到概率。",
            "3) 若模型既无 predict_proba 也无 decision_function，则返回 0 作为兜底概率。",
            "4) 加权融合：将五个模型的垃圾概率乘以各自权重求和，再除以权重总和，得到最终展示的垃圾邮件概率。",
            "",
        ])

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
