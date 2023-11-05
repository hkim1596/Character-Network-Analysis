import pandas as pd
import os
import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
import networkx as nx
import community as community_louvain
import plotly.graph_objs as go
import plotly.express as px

# Ensure output directories exist
os.makedirs('output_deep_clustering_kmeans', exist_ok=True)
os.makedirs('output_deep_clustering_community', exist_ok=True)

# Load the CSV file with the play titles
metadata_dir = "metadata"
plays_df = pd.read_csv(os.path.join(metadata_dir, "list_of_shakespeare_plays4.csv"), header=None)
play_title_mapping = dict(zip(plays_df[0].str.strip(), plays_df[1].str.strip()))

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate the output filename based on the CSV filename
def get_output_filename(json_filename, middle_suffix="", suffix="_deep_clustering.html"):
    # Get the base of the filename (without extension)
    base_filename = os.path.splitext(os.path.basename(json_filename))[0]
    
    # Get the full play title from the mapping, if no title mapping found, use original filename
    full_play_title = play_title_mapping.get(base_filename, base_filename)
    
    # Replace spaces with underscores if there are any spaces
    full_play_title = full_play_title.replace(" ", "_") if " " in full_play_title else full_play_title

    # Append the required suffix to the play title
    return f"{full_play_title}{middle_suffix}{suffix}"

# Function to get BERT embedding
def get_bert_embedding(text, max_length=510):
    # Tokenize and handle the text chunk by chunk
    tokens = tokenizer.tokenize(text)
    token_chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    
    # Get embeddings for chunks and aggregate
    embeddings = []
    for chunk in token_chunks:
        inputs = tokenizer.encode_plus(chunk, add_special_tokens=True, return_tensors='pt')
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[0][0].detach().numpy())
    return np.mean(embeddings, axis=0)

# Process each JSON file in the directory
for json_file in os.listdir('output_json'):
    if json_file.endswith('.json'):
        file_path = os.path.join("output_json", json_file)
        with open(file_path, "r") as file:
            data = [json.loads(line) for line in file]
        df = pd.DataFrame(data)

        # Convert words to embeddings and calculate average for each character
        character_embeddings = {char: get_bert_embedding(' '.join(df[df['speaker'] == char]['text'])) for char in df['speaker'].unique()}

        # Find optimal number of clusters using KMeans and Davies-Bouldin Index
        embeddings = list(character_embeddings.values())
        db_scores, K = [], range(2, len(character_embeddings))
        for k in tqdm(K, desc="Finding optimal clusters for " + json_file):
            kmeans = KMeans(n_clusters=k, n_init=10).fit(embeddings)
            db_scores.append(davies_bouldin_score(embeddings, kmeans.labels_))

        # Plotting Davies-Bouldin scores using Plotly
        db_output_filename = get_output_filename(json_file, prefix="DB_Index_")
        fig_db = px.line(x=K, y=db_scores, labels={'x':'Number of Clusters', 'y':'Davies-Bouldin Score'}, title='Davies-Bouldin Index Analysis')
        fig_db.write_html(os.path.join('output_deep_clustering_kmeans', db_output_filename))

        # t-SNE visualization
        optimal_clusters = K[db_scores.index(min(db_scores))]

        # Convert list of embeddings to a NumPy array
        embeddings_array = np.array(embeddings)

        n_samples = embeddings_array.shape[0]
        perplexity_value = min(30, n_samples - 1)  # Common default value for perplexity is 30, but it needs to be less than n_samples

        # Create and fit the t-SNE model
        tsne_model = TSNE(n_components=2, perplexity=perplexity_value, learning_rate='auto', init='pca', random_state=0)
        low_dim_embeddings = tsne_model.fit_transform(embeddings_array)

        df_plotly = pd.DataFrame({'x': low_dim_embeddings[:, 0], 'y': low_dim_embeddings[:, 1], 'label': list(character_embeddings.keys()), 'cluster': KMeans(n_clusters=optimal_clusters, n_init=10).fit_predict(embeddings)})
        fig_tsne = px.scatter(df_plotly, x='x', y='y', color='cluster', text='label', title="t-SNE visualization of character clusters")
        tsne_output_filename = get_output_filename(json_file, prefix="tSNE_")
        fig_tsne.write_html(os.path.join('output_deep_clustering_kmeans', tsne_output_filename))

        # Community detection with Louvain
        G = nx.from_pandas_edgelist(df, 'speaker', 'listener')
        partition = community_louvain.best_partition(G)

        # Create a Plotly scatter plot for community detection
        def plot_community(G, partition):
            pos = nx.spring_layout(G)
            communities = set(partition.values())
            community_colors = {node: partition[node] for node in G.nodes()}

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

            node_x = []
            node_y = []
            node_text = []
            node_color = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                node_color.append(community_colors[node])

            node_trace = go.Scatter(
                x=node_x, y=node_y, text=node_text,
                mode='markers+text',
                textposition="bottom center",
                hoverinfo='text',
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    color=node_color,
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title='Community',
                        xanchor='left',
                        titleside='right'
                    ),
                    line_width=2))

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0, l=0, r=0, t=0),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )
            return fig

        # Generate and save the community detection plot with Plotly
        community_fig = plot_community(G, partition)
        community_output_filename = get_output_filename(json_file, middle_suffix="_community")
        community_fig.write_html(os.path.join('output_deep_clustering_community', community_output_filename))


