import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
matrix_before_path = 'matrix_before_1.csv'
matrix_after_path = 'x2_after.csv'

matrix_before = pd.read_csv(matrix_before_path, index_col=0)
matrix_after = pd.read_csv(matrix_after_path, index_col=0)

# 将矩阵转换为浮点类型
matrix_before = matrix_before.astype(float)
matrix_after = matrix_after.astype(float)

# 创建一个新的矩阵来显示下三角和上三角部分
combined_matrix = matrix_before.copy()

# 保留 matrix_before 的下三角部分和对角线
combined_matrix.values[np.triu_indices_from(combined_matrix, 1)] = np.nan

# 保留 matrix_after 的上三角部分（不包含对角线）
for i in range(combined_matrix.shape[0]):
    for j in range(i + 1, combined_matrix.shape[1]):
        combined_matrix.iat[i, j] = matrix_after.iat[i, j]

# 绘制热力图
plt.figure(figsize=(14, 12))
sns.heatmap(combined_matrix, cmap='Blues', cbar=False, square=True, xticklabels=True, yticklabels=True)

# 添加标题
plt.title('Combined Protein Interaction Heatmap', fontsize=16)

# 显示图表
plt.show()