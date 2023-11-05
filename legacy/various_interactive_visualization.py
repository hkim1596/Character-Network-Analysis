import os
import pandas as pd
import networkx as nx
import community as community_louvain
import plotly.graph_objects as go

# Read all CSV files in the output_matrices directory
csv_files = [f for f in os.listdir('output_matrix') if f.endswith('.csv')]

for csv_file in csv_files:
    # Read the interaction matrix from the CSV file
    df = pd.read_csv(os.path.join('output_matrix', csv_file))
    interaction_matrix = df.drop(df.columns[0], axis=1).values
    character_names = df.columns[1:].tolist()  # Get all column names after the first one

    # Create a graph from the interaction matrix
    G = nx.from_numpy_array(interaction_matrix)
    labels = {idx: character for idx, character in enumerate(character_names)}

    # Network visualization
    pos = nx.spring_layout(G)
    
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
                title=f'Community Detection for {csv_file.split(".")[0]}',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.write_html(os.path.join('community_detection', f"{csv_file.split('.')[0]}_community.html"))

    # Heatmap Visualization with Plotly
    heatmap_fig = go.Figure(data=go.Heatmap(
                z=interaction_matrix,
                x=character_names,
                y=character_names,
                colorscale='Blues'))
    heatmap_fig.write_html(os.path.join('heatmaps', f"{csv_file.split('.')[0]}_heatmap.html"))

    