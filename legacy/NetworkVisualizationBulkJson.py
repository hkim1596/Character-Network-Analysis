import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import json

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the "output" directory
output_dir = os.path.join(script_directory, "output")
output_image_dir = os.path.join(script_directory, "output_image")

def remove_after_underscore(name):
    return name.split('_')[0]

def process_and_save_graph(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    exchanges = data["word_counts_summary"]
    
    # Remove the part after underscore
    df = pd.DataFrame(exchanges)
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
    pos = nx.spring_layout(G, k=0.5)  
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=100, edge_color='gray', width=0.1, font_size=5)
   
    # Extract play name from the filename and set as title
    play_name = os.path.splitext(os.path.basename(file_path))[0]
    plt.title(play_name)
    
    # Save the figure as a high-resolution PNG file inside the "output_image" folder
    output_name = play_name + '.png'
    output_path = os.path.join(output_image_dir, output_name)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

# Ensure output_image directory exists
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

for filename in os.listdir(output_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(output_dir, filename)
        process_and_save_graph(file_path)
