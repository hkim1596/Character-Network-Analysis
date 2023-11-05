import os
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# Ensure output directories exist
os.makedirs('community_detection', exist_ok=True)
os.makedirs('heatmaps', exist_ok=True)

# Read all CSV files in the output_matrices directory
csv_files = [f for f in os.listdir('output_matrix') if f.endswith('.csv')]

soliloquy_data = []  # List to store soliloquy data

for csv_file in csv_files:
    # Read the interaction matrix from the CSV file
    df = pd.read_csv(os.path.join('output_matrix', csv_file))
    interaction_matrix = df.drop(df.columns[0], axis=1).values
    character_names = df.columns[1:].tolist()  # Get all column names after the first one

    # Extract soliloquy values
    soliloquies = np.diag(interaction_matrix)
    
    # Extract play name from filename
    play_name = csv_file.split('.')[0]
    
    # Store the soliloquy data
    for char_name, words in zip(character_names, soliloquies):
        soliloquy_data.append({'Play': play_name, 'Character': char_name, 'Words': words})
    
    # Check if the matrix is square
    if interaction_matrix.shape[0] != interaction_matrix.shape[1]:
        print(f"Matrix in {csv_file} is not square! Shape: {interaction_matrix.shape}. Skipping this file.")
        continue

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
                title=f'Community Detection for {os.path.splitext(csv_file)[0]}',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.write_html(os.path.join('community_detection', f"{os.path.splitext(csv_file)[0]}_community.html"))

    
    # Heatmap Visualization with Plotly
    colorscale = [
        [0, 'rgb(255, 255, 255)'],       # color for value 0 (white)
        [1e-9, 'rgb(173, 216, 230)'],    # color just above 0 (light blue)
        [500/6000, 'rgb(100, 149, 237)'], # color around 500 (a darker shade of blue)
        [1, 'rgb(0, 0, 139)']            # color for the max value, 6000 (darkest blue)
    ]
    heatmap_fig = go.Figure(data=go.Heatmap(
                    z=np.flipud(interaction_matrix), # Flip the matrix vertically
                    x=character_names,
                    y=list(reversed(character_names)),
                    colorscale=colorscale,
                    zmin=0,
                    zmax=6000,
                    hovertemplate='Speaker: %{y}<br>Listener: %{x}<br>Words: %{z}<extra></extra>'
                ))

    heatmap_fig.update_layout(
        xaxis=dict(side='top')
    )

    # heatmap_fig = go.Figure(data=go.Heatmap(
    #             z=np.flipud(interaction_matrix), # Flip the matrix vertically
    #             x=character_names,
    #             y=list(reversed(character_names)), # Reverse the y-axis order
    #             colorscale='Electric'))
    heatmap_fig.write_html(os.path.join('heatmaps', f"{os.path.splitext(csv_file)[0]}_heatmap.html"))

# Convert the list of dictionaries into a DataFrame
soliloquy_df = pd.DataFrame(soliloquy_data)

# Filter out rows with 0 words
soliloquy_df = soliloquy_df[soliloquy_df['Words'] > 0]

# Group by 'Play' and sum the 'Words' to get total soliloquy words per play
play_totals = soliloquy_df.groupby('Play').agg({'Words': 'sum'}).rename(columns={'Words': 'TotalPlayWords'})

# Merge this aggregated data with the original soliloquy_df on the 'Play' column
merged_df = soliloquy_df.merge(play_totals, on='Play')

# Sort by 'TotalPlayWords' in descending order, then by 'Words' for individual characters
sorted_df = merged_df.sort_values(by=['TotalPlayWords', 'Words'], ascending=[False, False])

# Save this DataFrame to a new CSV
sorted_df.to_csv('sorted_aggregated_soliloquy_data.csv', index=False)

# Save this DataFrame to a new CSV
soliloquy_df.to_csv('aggregated_soliloquy_data.csv', index=False)

# Remove '_matrix' from the 'Play' column
sorted_df['Play'] = sorted_df['Play'].str.replace('_matrix', '')

# Create a new column combining 'Character' and 'PlayWithTotal' for the X-axis
sorted_df['Label'] = sorted_df['Character'] + ' in ' + sorted_df['Play'] + ' (' + sorted_df['TotalPlayWords'].astype(int).astype(str) + ' words)'

# Create the bar plot using Plotly Express for auto-coloring by play
fig = px.bar(sorted_df,
             x='Label',
             y='Words',
             color='Play',
             text='Words',
             color_discrete_sequence=px.colors.qualitative.Set1,  # Set color scheme
             title='Soliloquy Word Counts by Character',
             hover_data={'Label': False, 'Play': True, 'Character': True, 'Words': True},
             hover_name='Label'
            )

# Update the hover template
fig.update_traces(
    hovertemplate="<br>".join([
        "Play: %{customdata[0]}",
        "Character: %{customdata[1]}",
        "Words: %{y}"
    ])
)

# Update layout for better appearance
fig.update_layout(xaxis_title='Character in Play (Total words)',
                  yaxis_title='Words',
                  xaxis_tickangle=-45,  # Angle of X-axis labels for better readability
                  showlegend=False,
                  )

# Adjust the position of text on bars for better visibility
fig.update_traces(textposition='outside')

# Save to an HTML file
fig.write_html("bar_plot_soliloquies.html")

# Show the plot
fig.show()