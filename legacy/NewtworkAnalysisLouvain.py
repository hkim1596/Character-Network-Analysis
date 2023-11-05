import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import community
import matplotlib.cm as cm

# Load the data from the CSV file
df = pd.read_csv('word_counts_output.csv')

def remove_after_underscore(name):
    return name.split('_')[0]

# Remove the part after underscore for both speakers and listeners
df['speaker'] = df['speaker'].apply(remove_after_underscore)
df['listener'] = df['listener'].apply(remove_after_underscore)

# Create a new graph from the data
G = nx.Graph()

# Add edges to the graph
for _, row in df.iterrows():
    G.add_edge(row['speaker'], row['listener'], weight=row['total_words'])

# Compute the best partition using Louvain method
partition = community.best_partition(G)

# Visualize the graph with communities colored
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=2000, cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=15)
plt.show()

# Print the number of communities
num_communities = len(set(partition.values()))
print(f"There are {num_communities} communities.")
