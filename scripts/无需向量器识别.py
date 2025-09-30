import email
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import os

# 加载模型
MODEL_PATH = Path("E:/毕业设计/新测试/spam_classifier_model.joblib")  # 修改为你的模型路径

# 加载训练好的模型
model = joblib.load(MODEL_PATH)


# 解析邮件内容
def extract_email_content(email_path):
    """从 .eml 文件中提取邮件内容"""
    with open(email_path, 'r', encoding='utf-8', errors='ignore') as f:
        msg = email.message_from_file(f)

        # 检查邮件的编码和内容
        if msg.is_multipart():
            # 如果邮件是多部分的（例如带有附件），提取文本部分
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    return body
        else:
            body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            return body
    return ""


# 对单个邮件进行预测
def predict_email(model, email_path):
    """预测单个邮件是否为垃圾邮件"""
    email_content = extract_email_content(email_path)
    if email_content:
        # 使用模型直接进行预测
        # 将邮件内容转换为二维数组（必须是二维数组）
        email_content = np.array([email_content])  # 转换为二维数组 (1, N)
        prediction = model.predict(email_content)  # 直接预测邮件内容
        return prediction[0]  # 返回预测结果 (0: ham, 1: spam)
    return None


# 遍历目录中的所有邮件文件并进行预测
def predict_emails_in_directory(model, email_directory):
    """遍历目录中的邮件文件并进行预测"""
    result = []
    for email_file in Path(email_directory).rglob('*.eml'):  # 遍历所有 .eml 文件
        print(f"正在处理文件: {email_file}")
        prediction = predict_email(model, email_file)
        if prediction is not None:
            result.append((email_file.name, prediction))
            print(f"预测结果: {'垃圾邮件' if prediction == 1 else '正常邮件'}")
        else:
            print(f"无法处理文件: {email_file}")
    return result


# 主程序
def main():
    email_directory = "E:/毕业设计/邮件集/datacon2023-spoof-email-main/day1"  # 修改为你的邮件目录
    result = predict_emails_in_directory(model, email_directory)

    # 打印结果
    for email_name, prediction in result:
        print(f"邮件: {email_name}, 预测: {'垃圾邮件' if prediction == 1 else '正常邮件'}")


if __name__ == "__main__":
    main()
