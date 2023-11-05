import os
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import plotly.graph_objects as go

# Load the CSV file with the play titles
metadata_dir = "metadata"
plays_df = pd.read_csv(os.path.join(metadata_dir, "list_of_shakespeare_plays3.csv"), header=None)

# Create a mapping dictionary
play_title_mapping = dict(zip(plays_df[0], plays_df[1]))

# Function to generate the output filename based on the CSV filename
def get_output_filename(csv_filename):
    # Get the full play title from the mapping
    full_play_title = play_title_mapping.get(csv_filename)
    if not full_play_title:
        raise ValueError(f"No title mapping found for {csv_filename}")
    
    # Replace spaces with underscores if there are any spaces
    full_play_title = full_play_title.replace(" ", "_") if " " in full_play_title else full_play_title

    # Append the required suffix to the play title
    return f"{full_play_title}_exchange_community.html"

# Ensure output directory exists
os.makedirs('output_exchange_community', exist_ok=True)

# Read all CSV files in the output_matrices directory
csv_files = [f for f in os.listdir('output_exchange') if f.endswith('.csv')]

for csv_file in csv_files:
    csv_file_path = os.path.join('output_exchange', csv_file)  # This creates the correct path
    try:
        # Read the interaction matrix from the CSV file
        df = pd.read_csv(csv_file_path)
        interaction_matrix = df.drop(df.columns[0], axis=1).values
        character_names = df.columns[1:].tolist()  # Get all column names after the first one
        
        # Check if the matrix is square
        if interaction_matrix.shape[0] != interaction_matrix.shape[1]:
            print(f"Matrix in {csv_file} is not square! Shape: {interaction_matrix.shape}. Skipping this file.")
            continue

        # Create a graph from the interaction matrix
        G = nx.from_numpy_array(interaction_matrix)
        labels = {idx: character for idx, character in enumerate(character_names)}
        pos = nx.spring_layout(G)  # Position the nodes using the spring layout algorithm

        # Community detection (Louvain method)
        partition = community_louvain.best_partition(G)
        cmap = go.Figure().add_trace(go.Scatter()).data[0].marker.colorscale
        colors = [partition[node] for node in G.nodes()]

        # Network Visualization with Plotly
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

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[labels[node] for node in G.nodes()],
            textposition='top center',
            marker=dict(
                showscale=True,
                colorscale=cmap,
                color=colors,
                size=10,
                line_width=2,
                line=dict(color='black')
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f'Community Detection for {os.path.splitext(csv_file)[0]}',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0,l=0,r=0,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        
        # Generate the output filename using the function provided
        output_filename = get_output_filename(csv_file)
        fig.write_html(os.path.join('output_exchange_community', output_filename))
    except Exception as e:
            print(f"An error occurred while processing {csv_file}: {e}")
            continue   