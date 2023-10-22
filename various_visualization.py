import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import community as community_louvain

# Read all CSV files in the output_matrices directory
csv_files = [f for f in os.listdir('output_matrix') if f.endswith('.csv')]

for csv_file in csv_files:
    # Read the interaction matrix from the CSV file
    df = pd.read_csv(os.path.join('output_matrix', csv_file))
    
    # Drop any unnamed columns (like indices)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    interaction_matrix = df.values
    character_names = df.columns.tolist()

    # Ensure matrix is square
    if interaction_matrix.shape[0] != interaction_matrix.shape[1]:
        print(f"Matrix in {csv_file} is not square, skipping...")
        continue

    # Create a graph from the interaction matrix
    G = nx.from_numpy_array(interaction_matrix)
    labels = {idx: character for idx, character in enumerate(character_names)}

    # Network visualization
    pos = nx.spring_layout(G)
    # pos = nx.spiral_layout(G)
    # pos = nx.circular_layout(G)
    edge_widths = [G[u][v]['weight'] / 100.0 for u, v in G.edges()]

    # Save network visualization
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=700, node_color='lightblue', font_size=10, edge_color='gray', width=edge_widths)
    if not os.path.exists('network_visualizations'):
        os.mkdir('network_visualizations')
    plt.savefig(os.path.join('network_visualizations', f"{csv_file.split('.')[0]}_network.png"))
    plt.close()

    # Save community detection plot (Louvain method)
    partition = community_louvain.best_partition(G)

    # Save community detection plot
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.jet

    # Normalize the partition values for color mapping
    unique_communities = list(set(partition.values()))
    num_communities = len(unique_communities)
    normalized_partition = {node: unique_communities.index(value)/float(num_communities-1) for node, value in partition.items()}
    colors = [cmap(normalized_partition[node]) for node in G.nodes()]

    # Draw the nodes with the desired styles
    nx.draw_networkx_nodes(G, pos, node_color='none', edgecolors=colors, linewidths=2, node_size=300)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.2, alpha=0.5)
    # Adjust the vertical position of labels to place them above the nodes
    offset_y = 0.07  # adjust this value as needed
    label_pos = {node: (x, y+offset_y) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=5)

    if not os.path.exists('community_detection'):
        os.mkdir('community_detection')
    plt.savefig(os.path.join('community_detection', f"{csv_file.split('.')[0]}_community.png"), dpi=600)
    plt.close()

    
    # Heatmap visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix, annot=True, cmap="YlGnBu", xticklabels=character_names, yticklabels=character_names)
    if not os.path.exists('heatmaps'):
        os.mkdir('heatmaps')
    plt.savefig(os.path.join('heatmaps', f"{csv_file.split('.')[0]}_heatmap.png"))
    plt.close()
