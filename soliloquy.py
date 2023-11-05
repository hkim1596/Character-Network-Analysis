import os
import pandas as pd
import numpy as np
import plotly.express as px

# Function to extract soliloquy data from CSV files
def extract_soliloquy_data(csv_files):
    soliloquy_data = []  # List to store soliloquy data

    for csv_file in csv_files:
        df = pd.read_csv(os.path.join('output_exchange', csv_file))
        interaction_matrix = df.drop(df.columns[0], axis=1).values
        character_names = df.columns[1:].tolist()

        soliloquies = np.diag(interaction_matrix)
        play_name = csv_file.split('.')[0]

        for char_name, words in zip(character_names, soliloquies):
            soliloquy_data.append({'Play': play_name, 'Character': char_name, 'Words': words})
            
        # Additional code for checking matrix shape etc. can be placed here if required

    return soliloquy_data

# Read all CSV files in the output_matrices directory
csv_files = [f for f in os.listdir('output_exchange') if f.endswith('.csv')]

# Extract the soliloquy data
soliloquy_data = extract_soliloquy_data(csv_files)

# Convert the list of dictionaries into a DataFrame
soliloquy_df = pd.DataFrame(soliloquy_data)

# Filter out rows with 0 words
soliloquy_df = soliloquy_df[soliloquy_df['Words'] > 0]

# Group by 'Play' and sum the 'Words' to get total soliloquy words per play
play_totals = soliloquy_df.groupby('Play').agg({'Words': 'sum'}).rename(columns={'Words': 'TotalPlayWords'})

# Merge this aggregated data with the original soliloquy_df on the 'Play' column
merged_df = soliloquy_df.merge(play_totals, on='Play')

# Sort by 'TotalPlayWords' in descending order, then by 'Words' for individual characters
sorted_df = merged_df.sort_values(by=['TotalPlayWords', 'Words'], ascending=[False, False])

# Save this DataFrame to a new CSV
sorted_df.to_csv('sorted_aggregated_soliloquy_data.csv', index=False)

# Save this DataFrame to a new CSV
soliloquy_df.to_csv('aggregated_soliloquy_data.csv', index=False)

print("Soliloquy data extracted and saved to CSV.")

# Remove '_matrix' from the 'Play' column
sorted_df['Play'] = sorted_df['Play'].str.replace('_matrix', '')

# Create a new column combining 'Character' and 'PlayWithTotal' for the X-axis
sorted_df['Label'] = sorted_df['Character'] + ' in ' + sorted_df['Play'] + ' (' + sorted_df['TotalPlayWords'].astype(int).astype(str) + ' words)'

# Create the bar plot using Plotly Express for auto-coloring by play
fig = px.bar(sorted_df,
             x='Label',
             y='Words',
             color='Play',
             text='Words',
             color_discrete_sequence=px.colors.qualitative.Set1,  # Set color scheme
             title='Soliloquy Word Counts by Character',
             hover_data={'Label': False, 'Play': True, 'Character': True, 'Words': True},
             hover_name='Label'
            )

# Update the hover template
fig.update_traces(
    hovertemplate="<br>".join([
        "Play: %{customdata[0]}",
        "Character: %{customdata[1]}",
        "Words: %{y}"
    ])
)

# Update layout for better appearance
fig.update_layout(xaxis_title='Character in Play (Total words)',
                  yaxis_title='Words',
                  xaxis_tickangle=-45,  # Angle of X-axis labels for better readability
                  showlegend=False,
                  )

# Adjust the position of text on bars for better visibility
fig.update_traces(textposition='outside')

# Save to an HTML file
fig.write_html("bar_plot_soliloquies.html")
