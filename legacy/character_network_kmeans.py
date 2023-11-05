import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Check if 'output_kmeans' directory exists, if not, create it
if not os.path.exists('output_kmeans'):
    os.makedirs('output_kmeans')

# List all CSV files in the 'output_matrics' directory
csv_files = [f for f in os.listdir('output_matrix') if f.endswith('.csv')]

for file in csv_files:
    # Load the interaction matrix
    file_path = os.path.join('output_matrix', file)
    interaction_matrix = pd.read_csv(file_path, index_col=0)

    # Scaling and Preprocessing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(interaction_matrix)

    # Clustering with K-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    # Evaluation
    silhouette_avg = silhouette_score(scaled_data, clusters)
    print(f'Silhouette Score for {file}: {silhouette_avg}')

    # Visualization using PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)

    # Plotting the 2D data
    plt.figure(figsize=(12, 7))
    plt.scatter(principal_components[:, 0], principal_components[:, 1],
                c=clusters, cmap='rainbow', edgecolors='k', s=100)
    plt.title(f'2D PCA of Characters Clustering for {file}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()

    # Annotating data points with character names
    characters = interaction_matrix.index.tolist()
    for i, character in enumerate(characters):
        plt.annotate(character, (principal_components[i, 0], principal_components[i, 1]), fontsize=8)

    # Specify the path to save the figure in the 'output_kmeans' folder
    output_filename = os.path.splitext(file)[0] + '_kmeans.png'
    output_path = os.path.join('output_kmeans', output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close()

    print(f"Visualization saved for {file} in {output_path}")
