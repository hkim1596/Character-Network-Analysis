import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV file with the play titles
metadata_dir = "metadata"
plays_df = pd.read_csv(os.path.join(metadata_dir, "list_of_shakespeare_plays3.csv"), header=None)
# Create a mapping dictionary without replacing '.csv'
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
    return f"{full_play_title}_exchange_kmeans.html"


# Directories
input_directory = 'output_exchange'
output_directory = 'output_exchange_kmeans'

# Create output directory if it does not exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Initialize an empty list to collect silhouette scores
silhouette_scores = []

# Process each CSV file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_directory, filename)
        interaction_matrix = pd.read_csv(filepath, index_col=0)

    # Scaling and Preprocessing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(interaction_matrix)  # Ensure this line is present

    # Clustering with K-means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    # Evaluation
    silhouette_avg = silhouette_score(scaled_data, clusters)
    print(f'Silhouette Score for {filename}: {silhouette_avg}')

    # Append silhouette score and file name to the list
    silhouette_scores.append({'File': filename, 'Silhouette Score': silhouette_avg})

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
        title=f'2D PCA of Characters Clustering for {filename}',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2'
    )

    # Specify the path to save the figure in the 'output_kmeans' folder
    try:
            output_filename = get_output_filename(filename)
    except ValueError as e:
            print(e)
            continue
    output_path = os.path.join(output_directory, output_filename)
    fig.write_html(output_path)

    print(f"Visualization saved for {filename} in {output_path}")

# Convert the list of dictionaries to a DataFrame
df_scores = pd.DataFrame(silhouette_scores)

# Sort the DataFrame in ascending order based on Silhouette Score
df_scores = df_scores.sort_values(by='Silhouette Score', ascending=True)

# Specify the path to save the CSV file
csv_output_path = os.path.join(output_directory, 'silhouette_scores.csv')

# Write the DataFrame to a CSV file
df_scores.to_csv(csv_output_path, index=False)

# Create a bar graph using plotly.express
fig_bar = px.bar(df_scores, x='File', y='Silhouette Score', title='Silhouette Scores per File')
fig_bar.write_html(os.path.join(output_directory, 'silhouette_scores.html'))