import networkx as nx
import pandas as pd
import os

# Directory path
output_dir = 'output'
output_metrics_dir = 'output_metrics'
files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]

# If the output_metrics directory does not exist, create it
if not os.path.exists(output_metrics_dir):
    os.makedirs(output_metrics_dir)

# Dataframe to store results
results = pd.DataFrame(columns=['Play', 'Degree Centrality', 'Betweenness Centrality', 
                                'Closeness Centrality', 'Eigenvector Centrality', 
                                'Density', 'Cluster Coefficient'])

for f in files:
    # Load dataframe from CSV
    df = pd.read_csv(os.path.join(output_dir, f))
    
    # Create the graph from the dataframe
    G = nx.from_pandas_edgelist(df, 'speaker', 'listener', 'total_words')

    # Calculate metrics
    degree_centrality = max(nx.degree_centrality(G).values())
    betweenness_centrality = max(nx.betweenness_centrality(G).values())
    closeness_centrality = max(nx.closeness_centrality(G).values())
    eigenvector_centrality = max(nx.eigenvector_centrality(G).values())
    density = nx.density(G)
    cluster_coefficient = nx.average_clustering(G)

    # Append results to dataframe
    new_row = pd.DataFrame([{
        'Play': f.replace('.csv', ''),
        'Degree Centrality': degree_centrality,
        'Betweenness Centrality': betweenness_centrality,
        'Closeness Centrality': closeness_centrality,
        'Eigenvector Centrality': eigenvector_centrality,
        'Density': density,
        'Cluster Coefficient': cluster_coefficient
    }])
    
    results = pd.concat([results, new_row], ignore_index=True)

# Save results to a CSV in the 'output_metrics' folder
results.to_csv(os.path.join(output_metrics_dir, 'metrics_summary.csv'), index=False)
