import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'data/unique_new_protein_connections (1).csv'
data = pd.read_csv(file_path)

# Calculate the number of unique connections for each protein
protein_connections = pd.concat([data['Protein1'], data['Protein2']])
protein_connection_counts = protein_connections.value_counts()

# Convert the Series to a DataFrame for plotting
protein_connection_counts_df = protein_connection_counts.reset_index()
protein_connection_counts_df.columns = ['Protein', 'Changes in Connections']

# Define the color
vblue = (27/255, 161/255, 226/255)  # Convert RGB to matplotlib color format

# Plot the data
plt.figure(figsize=(20,25))
bars = plt.barh(protein_connection_counts_df['Protein'], protein_connection_counts_df['Changes in Connections'], color=vblue)

# Adjust the height of the bars to remove gaps
for bar in bars:
    bar.set_height(1.0)  # Set the height to 1 to remove gaps

# Add spacing between the bars to match the format of the provided image
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.xlabel('Number of Changes in Connections', fontsize=30)
plt.ylabel('Proteins', fontsize=30)
plt.title('Proteins with the Most Changed Connections', fontsize=30)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Save the updated plot to a file
# output_path_vblue = 'most_changed_connections_vblue.png'
# plt.savefig(output_path_vblue)
