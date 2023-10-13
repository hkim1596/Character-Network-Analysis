import pandas as pd
import matplotlib.pyplot as plt

# Load the data from metrics_summary.csv
results = pd.read_csv('output_metrics/metrics_summary.csv')

# List of metrics to plot
metrics = ['Degree Centrality', 'Betweenness Centrality', 'Eigenvector Centrality', 'Closeness Centrality']

for metric in metrics:
    # Sort the results DataFrame based on the metric in ascending order
    sorted_results = results.sort_values(by=metric)
    
    plt.figure(figsize=(12, 6))
    
    # Bar plot for each metric
    sorted_results.plot(x='Play', y=metric, kind='bar', legend=False)
    
    plt.title(f'{metric} across Plays')
    plt.ylabel(metric)
    plt.xlabel('Plays')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(f'output_image/{metric} Bar Graph.png', dpi=600)
    #plt.show()
