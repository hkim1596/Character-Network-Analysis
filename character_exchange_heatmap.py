import os
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

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
    return f"{full_play_title}_exchange_heatmap.html"

# Ensure output directories exist
os.makedirs('output_exchange_heatmap', exist_ok=True)

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
            title=f"Heatmap of Interactions in {os.path.splitext(csv_file)[0]}",
            xaxis=dict(side='top'),
            yaxis=dict(autorange='reversed')  # Ensure the y-axis is correctly flipped
        )

        # Generate the output filename using the function provided
        output_filename = get_output_filename(csv_file)

        # Write the heatmap to an HTML file using the generated filename
        heatmap_fig.write_html(os.path.join('output_exchange_heatmap', output_filename)) 
    except Exception as e:
        print(f"An error occurred while processing {csv_file}: {e}")
        continue   