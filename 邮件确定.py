# 读取标签文件并打印
with open('E:/毕业设计/trec06c/full/index', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 打印所有标签，检查标签是否正确
for line in lines[:10]:  # 打印前 10 行
    print(line.strip())
