import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm


# Determine the current working directory
current_dir = os.getcwd()

# Construct the path to the JSON file in the output directory
file_path = os.path.join(current_dir, "output", "Ham.json")


# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load your JSON file using the constructed path
with open(file_path, "r") as file:
    data = json.load(file)
    df = pd.DataFrame(data["individual_exchanges"])

def get_bert_embedding(text):
    tokens = tokenizer.tokenize(text)
    
    # Truncate or pad the token sequence to fit within BERT's max length
    if len(tokens) > 510:
        tokens = tokens[:510]
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    # Convert tokens to IDs and ensure it's within BERT's max length
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = token_ids[:512]
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)

    with torch.no_grad():
        outputs = model(token_ids_tensor)
        return outputs[0][0, 0, :].numpy()


# Convert words exchanged to embeddings
df['interaction_embedding'] = df['text'].apply(get_bert_embedding)

# Aggregate embeddings for each character (mean of all interactions they are involved in)
characters = pd.concat([df['speaker'], df['listener']]).unique()
character_embeddings = {char: [] for char in characters}

for index, row in df.iterrows():
    embedding = row['interaction_embedding']
    character_embeddings[row['speaker']].append(embedding)
    character_embeddings[row['listener']].append(embedding)

for char, embeds in character_embeddings.items():
    character_embeddings[char] = sum(embeds) / len(embeds)

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

# Plotting Davies-Bouldin scores
plt.plot(K, db_scores, 'bx-')
plt.xlabel('k')
plt.ylabel('Davies-Bouldin values')
plt.title('Davies-Bouldin Index Analysis')
plt.show()

# Choosing the optimal number of clusters (the one with the lowest DB index)
optimal_clusters = K[db_scores.index(min(db_scores))]
print(f"Optimal number of clusters: {optimal_clusters}")

# Run KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(embeddings)
character_clusters = dict(zip(characters, kmeans.labels_))
