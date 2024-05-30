import matplotlib.pyplot as plt
import numpy as np

# 定义颜色
vblue = (27/255, 161/255, 226/255)
vgreen = (51/255, 153/255, 51/255)
vred = (229/255, 20/255, 0/255)
vpink = (216/255, 0/255, 115/255)

# 数据
methods = ['Accuracy', 'Recall', 'Precision', 'F1', 'MCC']
weightedPPI_diffusion = [0.85, 0.65, 0.9, 0.85, 0.75]
weightedPPI = [0.83, 0.6, 0.88, 0.83, 0.7]
PPI_diffusion = [0.8, 0.58, 0.87, 0.82, 0.65]
PPI = [0.75, 0.55, 0.85, 0.8, 0.6]

x = np.arange(len(methods))  # 标签位置
width = 0.2  # 条形图宽度

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width*1.5, weightedPPI_diffusion, width, label='weightedPPI+diffusion', color=vblue)
rects2 = ax.bar(x - width/2, weightedPPI, width, label='weightedPPI', color=vred)
rects3 = ax.bar(x + width/2, PPI_diffusion, width, label='PPI+diffusion', color=vgreen)
rects4 = ax.bar(x + width*1.5, PPI, width, label='PPI', color=vpink)

# 调整y轴起点
ax.set_ylim(0.4, 1.0)

# 添加标签、标题等
ax.set_ylabel('Score')
ax.set_xlabel('Metric')
ax.set_title('Comparison of Different Methods')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

fig.tight_layout()

# 保存图片
plt.savefig('comparison_of_methods.png')
plt.show()
