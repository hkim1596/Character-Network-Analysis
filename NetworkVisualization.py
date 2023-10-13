import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


# Load the data from the CSV file
df = pd.read_csv('word_counts_output.csv')

def remove_after_underscore(name):
    return name.split('_')[0]

# Remove the part after underscore
df['speaker'] = df['speaker'].apply(remove_after_underscore)
df['listener'] = df['listener'].apply(remove_after_underscore)

# Create a new graph from the data
G = nx.Graph()

# Add edges to the graph
for _, row in df.iterrows():
    G.add_edge(row['speaker'], row['listener'], weight=row['total_words']) # I've changed 'words' to 'total_words' since that's the column name in your grouped dataframe.

# Visualize the graph
plt.figure(figsize=(10, 10))
#pos = nx.spring_layout(G)  # Using spring layout
#pos = nx.kamada_kawai_layout(G) # Using the Kamada-Kawai layout

#nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', width=1.0, font_size=15)
#plt.show()

pos = nx.spring_layout(G, k=0.5)  # Increase k value for more spacing

nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', width=1.0, font_size=15)
#plt.show()

# Save the figure as a high-resolution PNG file inside the "output" folder
plt.savefig('output_image/network_visualization.png', dpi=600, bbox_inches='tight')
