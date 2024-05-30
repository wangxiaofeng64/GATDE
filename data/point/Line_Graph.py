import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load data from CSV
df = pd.read_csv('data/line_chart_data.csv')

# Set font
plt.rcParams['font.sans-serif'] = ['SimSun', 'Arial']

# Plot the data
sns.lineplot(x='x', y='Accuracy', data=df, marker='o', linestyle='-', color='b', label='Accuracy')
sns.lineplot(x='x', y='Recall', data=df, marker='s', linestyle='--', color='g', label='Recall')
sns.lineplot(x='x', y='F1 Score', data=df, marker='^', linestyle=':', color='r', label='F1 Score')
sns.lineplot(x='x', y='Precision', data=df, marker='p', linestyle='-.', color='c', label='Precision')
# sns.lineplot(x='x', y='Specificity', data=df, marker='d', linestyle='-', color='m', label='Specificity')
sns.lineplot(x='x', y='MCC', data=df, marker='h', linestyle='--', color='y', label='MCC')

# Remove right and top borders
plt.grid(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Add titles and labels
plt.xlabel('$\overline{r}$')
plt.ylabel('Score')
# Show the plot
plt.show()
