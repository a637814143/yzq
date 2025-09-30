import numpy as np
import joblib

# 加载 .npy 文件
vectorizer = np.load(r"E:\毕业设计\新测试\email_features.npy", allow_pickle=True)

# 保存为 .joblib 文件
joblib.dump(vectorizer, r"E:\毕业设计\新测试\email_features.joblib")
