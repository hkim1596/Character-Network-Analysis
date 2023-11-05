import pandas as pd
import plotly.express as px
import os

# Load the data from metrics_summary.csv
results = pd.read_csv('output_metrics/metrics_summary.csv')

# Ensure output directory exists
os.makedirs('output_centrality_bar_graphs', exist_ok=True)

# List of metrics to plot
metrics = ['Degree Centrality', 'Betweenness Centrality', 'Eigenvector Centrality', 'Closeness Centrality']

for metric in metrics:
    # Sort the results DataFrame based on the metric in descending order
    sorted_results = results.sort_values(by=metric, ascending=False)

    # Bar plot for each metric using Plotly
    fig = px.bar(sorted_results, x='Play', y=metric, title=f'{metric} across Plays')

    # Update layout
    fig.update_layout(
        xaxis_title="Play",
        yaxis_title=metric,
        xaxis={'categoryorder':'total descending'},
        title=dict(text=f'{metric} across Plays', x=0.5),
        autosize=True
    )
    
    # Customize x-axis tick angle
    fig.update_xaxes(tickangle=45)
    
    # Save the figure as an interactive HTML file
    fig.write_html(f'output_centrality_bar_graphs/{metric}_Bar_Graph.html')
