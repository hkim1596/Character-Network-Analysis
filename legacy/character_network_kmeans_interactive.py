import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
import plotly.express as px
import plotly.graph_objects as go

# Check if 'output_kmeans' directory exists, if not, create it
if not os.path.exists('output_kmeans'):
    os.makedirs('output_kmeans')

# List all CSV files in the 'output_matrix' directory
csv_files = [f for f in os.listdir('output_matrix') if f.endswith('.csv')]

for file in csv_files:
    # Load the interaction matrix
    file_path = os.path.join('output_matrix', file)
    interaction_matrix = pd.read_csv(file_path, index_col=0)

    # Scaling and Preprocessing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(interaction_matrix)

    # Clustering with K-means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    # Evaluation
    silhouette_avg = silhouette_score(scaled_data, clusters)
    print(f'Silhouette Score for {file}: {silhouette_avg}')

    # Visualization using PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)

    # Create a dataframe for plotly
    df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
    df['Cluster'] = clusters
    df['Characters'] = interaction_matrix.index

    # Plotting the 2D data using plotly.graph_objects
    fig = go.Figure()

    # Scatter plot of the data
    fig.add_trace(go.Scatter(
        x=df['PC1'],
        y=df['PC2'],
        mode='markers+text',
        marker=dict(
            color=df['Cluster'],
            colorscale='Rainbow',
            size=10,
            showscale=False  # this hides the colorbar
        ),
        text=df['Characters'],  # this will display character names
        textposition='top center'
    ))

    fig.update_layout(
        title=f'2D PCA of Characters Clustering for {file}',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2'
    )

    # Specify the path to save the figure in the 'output_kmeans' folder
    output_filename = os.path.splitext(file)[0] + '_kmeans.html'
    output_path = os.path.join('output_kmeans', output_filename)

    fig.write_html(output_path)

    print(f"Visualization saved for {file} in {output_path}")
