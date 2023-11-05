import os
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import plotly.graph_objects as go
import plotly.express as px


def compute_similarity(char1, char2):
    return sum((df.loc[char1].astype(bool)) & (df.loc[char2].astype(bool)))

input_directory = 'output_onstage'
output_directory = 'output_onstage_community'

metadata_dir = "metadata"
plays_df = pd.read_csv(os.path.join(metadata_dir, "list_of_shakespeare_plays2.csv"), header=None)
play_title_mapping = dict(zip(plays_df[0], plays_df[1]))

def get_output_filename(csv_filename):
    # Get the full play title from the mapping
    full_play_title = play_title_mapping.get(csv_filename, "").replace(" ", "_")
    if not full_play_title:
        raise ValueError(f"No title mapping found for {csv_filename}")
    # Append the required suffix to the play title
    return f"{full_play_title}_onstage_community.html"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(input_directory):
    if filename.endswith("_onstage.csv"):
        filepath = os.path.join(input_directory, filename)
        df = pd.read_csv(filepath, index_col=0)
        
        # Load data
        df = pd.read_csv(filepath, index_col=0)

        # Create graph from similarity matrix
        G = nx.Graph()
        for i, char1 in enumerate(df.index):
            for j, char2 in enumerate(df.index):
                weight = compute_similarity(char1, char2)
                if weight > 0:
                    G.add_edge(char1, char2, weight=weight)
        
        # Detect communities
        partition = community_louvain.best_partition(G, resolution=1.0)

        # Convert partition to colors for visualization
        community_numbers = list(set(partition.values()))
        colors = px.colors.qualitative.Set3
        color_map = {}
        for i, com in enumerate(community_numbers):
            color_map[com] = colors[i % len(colors)]
        
        node_colors = [color_map[partition[node]] for node in G.nodes()]

        # Visualize communities using NetworkX and Plotly
        pos = nx.spring_layout(G)
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = [pos[k][0] for k in G.nodes()]
        node_y = [pos[k][1] for k in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                colorscale='Rainbow',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2,
                color=node_colors
            )
        )
        
        node_trace.text = list(G.nodes())

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Character Network and Communities',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=0),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        
        # Save to HTML
        output_filename = get_output_filename(filename)  # Modified to use the new function
        output_filepath = os.path.join(output_directory, output_filename)
        fig.write_html(output_filepath)

print("All files processed and saved to", output_directory)
