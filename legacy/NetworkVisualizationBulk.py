import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def remove_after_underscore(name):
    return name.split('_')[0]

def process_and_save_graph(file_path):
    # Load the data from the CSV file
    df = pd.read_csv(file_path)

    # Remove the part after underscore
    df['speaker'] = df['speaker'].apply(remove_after_underscore)
    df['listener'] = df['listener'].apply(remove_after_underscore)

    # Create a new graph from the data
    G = nx.Graph()

    # Add edges to the graph
    for _, row in df.iterrows():
        G.add_edge(row['speaker'], row['listener'], weight=row['total_words'])

    # Determine character with the most words
    most_words_character = df.groupby('speaker').sum(numeric_only=True)['total_words'].idxmax()
    
    # Color map
    color_map = ['red' if node == most_words_character else 'skyblue' for node in G.nodes()]

    # Visualize the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.5)  # Increase k value for more spacing
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=100, edge_color='gray', width=0.1, font_size=5)
    
    # Extract play name from the filename (without '.csv') and set as title
    play_name = os.path.splitext(os.path.basename(file_path))[0]
    plt.title(play_name.replace('.csv', ''))

    # Save the figure as a high-resolution PNG file inside the "output_image" folder
    output_name = play_name + '.png'
    output_path = os.path.join('output_image', output_name)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

# Ensure output_image directory exists
if not os.path.exists('output_image'):
    os.makedirs('output_image')

# Iterate over each CSV file in the "output" folder
folder_path = 'output'
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        process_and_save_graph(file_path)
