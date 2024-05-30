import pandas as pd
import matplotlib.pyplot as plt

attention_weights_df = pd.read_csv('data/attention_weights.csv')

# Plot the data with the specified color and no gaps between bars
plt.figure(figsize=(20,25))
vblue = (27/255, 161/255, 226/255)  # Convert RGB to matplotlib color format
bars = plt.barh(attention_weights_df['Protein'], attention_weights_df['Attention Weight'], color=vblue)

# Adjust the height of the bars to remove gaps
for bar in bars:
    bar.set_height(1.0)  # Set the height to 1 to remove gaps

plt.xlabel('Attention Weight', fontsize=30)
plt.ylabel('Proteins', fontsize=30)
plt.title('Proteins by Attention Weight', fontsize=40)
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.subplots_adjust(top=1, bottom=0)  # Adjust subplots to reduce whitespace
plt.tight_layout()

# Save the updated plot to a file
output_path_attention_weights = 'proteins_by_attention_weights_1.png'
plt.show()
