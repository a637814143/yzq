import numpy as np
import joblib
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# 加载特征和标签数据
def load_features_and_labels(features_npy: str, labels_txt: str) -> tuple:
    """加载 .npy 文件中的特征和 .txt 文件中的标签"""

    # 加载特征
    X = np.load(features_npy)

    # 加载标签数据 (spam=1, ham=0)
    with open(labels_txt, 'r', encoding='utf-8') as f:
        y = [1 if line.strip().split()[0] == 'spam' else 0 for line in f.readlines()]

    # 检查标签数据的唯一值
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        raise ValueError(f"标签文件中只有一个类别的数据，无法训练模型。唯一标签值: {unique_labels}")

    return X, np.array(y)


# 训练模型并保存
def run_training_from_npy(features_npy: str, labels_txt: str, model_output: str | None = None):
    """从 .npy 和 .txt 文件加载数据并训练模型"""

    # 加载特征和标签
    X, y = load_features_and_labels(features_npy, labels_txt)

    # 将数据集划分为训练集和验证集（80% 训练，20% 验证）
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化逻辑回归模型
    model = LogisticRegression(max_iter=1000, solver='lbfgs')

    # 训练模型
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_valid)
    print("模型评估：")
    print(classification_report(y_valid, y_pred))
    print("混淆矩阵：")
    print(confusion_matrix(y_valid, y_pred))

    # 保存模型
    if model_output:
        joblib.dump(model, model_output)
        print(f"模型已保存至: {model_output}")
    else:
        print("模型训练完成，但未保存。")

    return model


# 主函数
def main():
    # 设置特征文件路径、标签文件路径以及模型保存路径
    features_npy = r"E:\毕业设计\新测试\email_features.npy"  # 特征文件路径
    labels_txt = r"E:\毕业设计\trec06c\full\index"  # 标签文件路径
    model_output_path = r"E:\毕业设计\新测试\spam_classifier_model.joblib"  # 模型保存路径

    try:
        # 运行训练并保存模型
        model = run_training_from_npy(
            features_npy=features_npy,
            labels_txt=labels_txt,
            model_output=model_output_path,
        )

        print(f"模型已训练完成，保存至 {model_output_path}")

    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


