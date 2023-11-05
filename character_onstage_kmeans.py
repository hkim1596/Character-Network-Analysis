import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import MDS
import plotly.express as px

# Function to compute similarity between characters
def compute_similarity(char1, char2, df):
    return sum((df.loc[char1].astype(bool)) & (df.loc[char2].astype(bool)))

# Load the CSV file with the play titles
metadata_dir = "metadata"
plays_df = pd.read_csv(os.path.join(metadata_dir, "list_of_shakespeare_plays.csv"), header=None)
# Create a mapping dictionary { 'H8_onstage.csv': 'Henry VIII', 'Ham_onstage.csv': 'Hamlet', ... }
play_title_mapping = dict(zip(plays_df[0].str.replace('.xml', '') + '_onstage.csv', plays_df[1]))

# Function to generate the output filename based on the CSV filename
def get_output_filename(csv_filename):
    # Get the full play title from the mapping
    full_play_title = play_title_mapping.get(csv_filename, "").replace(" ", "_")
    if not full_play_title:
        raise ValueError(f"No title mapping found for {csv_filename}")
    # Append the required suffix to the play title
    return f"{full_play_title}_onstage_kmeans.html"

# Directories
input_directory = 'output_onstage'
output_directory = 'output_onstage_kmeans'

# Create output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Process each CSV file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith("_onstage.csv"):
        filepath = os.path.join(input_directory, filename)
        df = pd.read_csv(filepath, index_col=0)
        
        # Compute similarity matrix
        similarity_matrix = np.zeros((len(df), len(df)))
        for i, char1 in enumerate(df.index):
            for j, char2 in enumerate(df.index):
                similarity_matrix[i, j] = compute_similarity(char1, char2, df)

        # Hierarchical clustering
        link = linkage(1 - similarity_matrix, method='complete')
        clusters = fcluster(link, 2, criterion='maxclust')

        # Dimensionality reduction for 2D visualization
        embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=False)
        transformed = embedding.fit_transform(1 - similarity_matrix)

        # Plotly visualization
        fig = px.scatter(x=transformed[:, 0], y=transformed[:, 1], color=clusters, text=df.index)
        fig.update_traces(marker=dict(size=15, opacity=0.8, line=dict(width=2, color='DarkSlateGrey')),
                          textposition='top center',
                          selector=dict(mode='markers+text'))
        
         # Generate the output filename using the mapping
        try:
            output_filename = get_output_filename(filename)
        except ValueError as e:
            print(e)
            continue
        
        output_filepath = os.path.join(output_directory, output_filename)
        fig.write_html(output_filepath)

# Print message when all files are processed
print("All files processed and saved to", output_directory)