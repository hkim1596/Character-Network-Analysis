def spring_layout_3d(G, iterations=50):
    pos = nx.spring_layout(G, dim=3)
    for _ in range(iterations):
        for node in G.nodes():
            # calculate the sum of the forces
            delta = np.zeros(3)
            for neighbor in G[node]:
                delta += pos[neighbor] - pos[node]
            norm = np.linalg.norm(delta)
            if norm > 0:
                delta /= norm
            pos[node] += delta * 0.04
    return pos


import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import community

# ... [same code for loading data and preprocessing]
# Load the data from the CSV file
df = pd.read_csv('word_counts_output.csv')

# Function to remove after underscore
def remove_after_underscore(name):
    return name.split('_')[0]

# Apply the function
df['speaker'] = df['speaker'].apply(remove_after_underscore)
df['listener'] = df['listener'].apply(remove_after_underscore)

# Create a new graph from the data
G = nx.Graph()

# Add edges to the graph
for _, row in df.iterrows():
    G.add_edge(row['speaker'], row['listener'], weight=row['total_words'])

# Compute the best partition using Louvain method
partition = community.best_partition(G)

# 3D visualization
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Get a list of unique groups
groups = set(partition.values())
mapping = dict(zip(sorted(groups), range(len(groups))))
colors = plt.cm.jet(np.linspace(0, 1, len(groups)))

# Updated position calculation using our 3D spring layout
pos = spring_layout_3d(G)

for node in G.nodes():
    ax.scatter(pos[node][0], pos[node][1], pos[node][2], s=100, c=[colors[mapping[partition[node]]]], label=str(node))
    ax.text(pos[node][0], pos[node][1], pos[node][2], s=node, fontsize=8)

ax.view_init(elev=20, azim=60)  # adjust the viewing angle for better visualization
plt.show()
