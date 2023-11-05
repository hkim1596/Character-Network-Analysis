import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS
import plotly.express as px

def compute_similarity(char1, char2):
    return sum((df.loc[char1].astype(bool)) & (df.loc[char2].astype(bool)))

input_directory = 'output_timeline'
output_directory = 'output_timeline_cluster'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_directory, filename)
        
        # Load data
        df = pd.read_csv(filepath, index_col=0)
        
        # Compute similarity matrix
        similarity_matrix = np.zeros((len(df), len(df)))
        for i, char1 in enumerate(df.index):
            for j, char2 in enumerate(df.index):
                similarity_matrix[i, j] = compute_similarity(char1, char2)

        # Hierarchical clustering
        link = linkage(1 - similarity_matrix, method='complete')
        clusters_2 = fcluster(link, 2, criterion='maxclust')

        # Dimensionality reduction for 2D visualization
        embedding = MDS(n_components=2, dissimilarity='precomputed', normalized_stress=False)
        transformed = embedding.fit_transform(1 - similarity_matrix)

        # Plotly visualization
        fig = px.scatter(x=transformed[:, 0], y=transformed[:, 1], color=clusters_2, text=df.index)
        fig.update_traces(marker=dict(size=15, opacity=0.8, line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers+text'))
        
        # Save to HTML
        output_filename = os.path.splitext(filename)[0] + "_cluster.html"
        output_filepath = os.path.join(output_directory, output_filename)
        fig.write_html(output_filepath)

print("All files processed and saved to", output_directory)
