import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import json
import os
from tqdm import tqdm
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE

# Determine the current working directory
current_dir = os.getcwd()

# Construct the path to the JSON file in the output directory
file_path = os.path.join(current_dir, "output", "Ham.json")

# Determine the path to the output_image2 directory
output_image2_dir = os.path.join(current_dir, "output_image2")
if not os.path.exists(output_image2_dir):
    os.makedirs(output_image2_dir)

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load your JSON file using the constructed path
with open(file_path, "r") as file:
    data = json.load(file)
    df = pd.DataFrame(data["individual_exchanges"])

def get_bert_embedding(text, max_length=510):
    # Split the text into chunks of max_length tokens
    tokens = tokenizer.tokenize(text)
    token_chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    
    embeddings = []
    for chunk in token_chunks:
        chunk = ['[CLS]'] + chunk + ['[SEP]']
        token_ids = tokenizer.convert_tokens_to_ids(chunk)
        token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(token_ids_tensor)
            embeddings.append(outputs[0][0, 0, :].numpy())
            
    # Aggregate the chunk embeddings by taking their mean
    embedding = np.mean(embeddings, axis=0)
    return embedding
# Convert words exchanged to embeddings
df['interaction_embedding'] = df['text'].apply(get_bert_embedding)

# Aggregate embeddings for each character (mean of all interactions they are involved in)
characters = pd.concat([df['speaker'], df['listener']]).unique()
# Convert words exchanged to embeddings and aggregate embeddings for each character
character_embeddings = {char: [] for char in characters}

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing interactions"):
    embedding = get_bert_embedding(row['text'])
    character_embeddings[row['speaker']].append(embedding)
    character_embeddings[row['listener']].append(embedding)

# Average the embeddings for each character
for char, embeds in character_embeddings.items():
    if embeds:  # Check if the list is not empty
        character_embeddings[char] = np.mean(embeds, axis=0)
    else:
        character_embeddings[char] = np.zeros((768,))

embeddings = list(character_embeddings.values())

# Determine the optimal number of clusters using Davies-Bouldin Index
db_scores = []
K = range(2, len(characters))

db_scores = []
K = range(2, len(characters))

# Add tqdm to show progress
for k in tqdm(K, desc="Processing KMeans"):
    kmeans = KMeans(n_clusters=k, n_init=10).fit(embeddings)
    labels = kmeans.labels_
    db_scores.append(davies_bouldin_score(embeddings, labels))
# Plotting Davies-Bouldin scores using Plotly
fig_db = px.line(x=K, y=db_scores, labels={'x':'k', 'y':'Davies-Bouldin values'}, title='Davies-Bouldin Index Analysis')
output_path_db = os.path.join(output_image2_dir, "Davies-Bouldin Index Analysis.html")
fig_db.write_html(output_path_db)

# Choosing the optimal number of clusters (the one with the lowest DB index)
optimal_clusters = K[db_scores.index(min(db_scores))]
print(f"Optimal number of clusters: {optimal_clusters}")

# Run KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(embeddings)
character_clusters = dict(zip(characters, kmeans.labels_))

# Convert the embeddings list into a NumPy array
embeddings_array = np.array(embeddings)

# Use t-SNE for dimensionality reduction with the array
tsne_model = TSNE(perplexity=30, n_components=2, init="pca", n_iter=3500, random_state=23)
low_dim_embeddings = tsne_model.fit_transform(embeddings_array)

# Get the cluster labels for each character
cluster_labels = kmeans.labels_

# Create a DataFrame for Plotly
df_plotly = pd.DataFrame({'x': low_dim_embeddings[:, 0], 'y': low_dim_embeddings[:, 1], 'label': characters, 'cluster': cluster_labels})
fig_tsne = px.scatter(df_plotly, x='x', y='y', color='cluster', text='label', title="t-SNE visualization of character clusters")
output_path_tsne = os.path.join(output_image2_dir, "tsne_visualization.html")
fig_tsne.write_html(output_path_tsne)
