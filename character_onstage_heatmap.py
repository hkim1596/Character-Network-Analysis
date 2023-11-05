import os
import pandas as pd
import plotly.graph_objects as go
from bs4 import BeautifulSoup

# Load the CSV file with the play titles
metadata_dir = "metadata"
plays_df = pd.read_csv(os.path.join(metadata_dir, "list_of_shakespeare_plays.csv"), header=None)

# Create a mapping dictionary { 'Tro.xml': 'Troilus and Cressida', 'H5.xml': 'Henry V', ... }
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
    return f"{full_play_title}_onstage_heatmap.html"

def process_xml_to_heatmap(file_path):
    # Read and Parse the XML content using BeautifulSoup
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    soup = BeautifulSoup(content, 'lxml-xml')

    # Initialize current narrative time, on_stage list and a dictionary to record characters on stage at each time point
    current_time = "0"
    on_stage = []
    stage_timeline = {}

    # Iterate through all nodes
    for tag in soup.find_all(['sp', 'stage', 'milestone']):
        # Update current time from milestone using xml:id attribute
        if tag.name == 'milestone' and tag.get('unit') == 'ftln':
            current_time = tag['xml:id']
            #print(f"Current Time (ftln): {current_time}. Characters on stage: {', '.join(on_stage) if on_stage else 'None'}")
            # Record characters on stage for the current time
            stage_timeline[current_time] = on_stage.copy()
        
        # Stage direction for entrance
        if tag.name == 'stage' and tag.get('type') == 'entrance' and 'who' in tag.attrs:
            characters_entering = [char for char in tag['who'].replace("#", "").split()]
            on_stage = list(set(on_stage + characters_entering))
        
        # Stage direction for exit
        if tag.name == 'stage' and tag.get('type') == 'exit' and 'who' in tag.attrs:
            characters_exiting = [char for char in tag['who'].replace("#", "").split()]
            on_stage = [char for char in on_stage if char not in characters_exiting]

    # At this point, the stage_timeline dictionary will contain the list of characters on stage for each current_time

    # Create a set of all unique characters that ever appear on stage
    all_characters = set()
    for char_list in stage_timeline.values():
        for char in char_list:
            all_characters.add(char)

    # Convert the set to a sorted list
    all_characters = sorted(list(all_characters))

    # Create an empty DataFrame with rows of all characters and columns of all the current_time values
    df = pd.DataFrame(index=all_characters, columns=stage_timeline.keys(), dtype=int)

    # Fill in the DataFrame
    for current_time, characters_on_stage in stage_timeline.items():
        for char in all_characters:
            if char in characters_on_stage:
                df.loc[char, current_time] = 1
            else:
                df.loc[char, current_time] = 0

    print(df)

    # Save the DataFrame to CSV file
    csv_output_directory = "output_onstage"
    if not os.path.exists(csv_output_directory):
        os.makedirs(csv_output_directory)

    csv_output_file_name = os.path.basename(file_path).replace(".xml", "_onstage.csv")
    csv_output_file_path = os.path.join(csv_output_directory, csv_output_file_name)
    df.to_csv(csv_output_file_path)

    # Modify x labels
    x_labels = [label.replace("ftln-", "TLN ") for label in df.columns]
    
    # Extract the suffix from filename for y labels modification
    file_suffix = "_" + os.path.basename(file_path).replace(".xml", "")
    
    # Modify y labels
    y_labels = [label.replace(file_suffix, "").title() for label in df.index]

    # Data for heatmap
    z = df.values
    x = x_labels
    y = y_labels

    hovertext = []

    for y_label, row in zip(y_labels, z):
        hover_row = []
        for x_label, val in zip(x_labels, row):
            tln = x_label.split()[-1]  # Assuming the format is 'TLN xxx'
            presence = 'Yes' if val == 1 else 'No'
            hover_data = f"Through Line Number: {tln}<br>Character: {y_label}<br>Presence: {presence}"
            hover_row.append(hover_data)
        hovertext.append(hover_row)

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        hovertext=hovertext,
        hoverinfo='text',  # Use custom hovertext
        colorscale='Blues',
        showscale=False
    ))

    # Update the layout
    fig.update_layout(
        autosize=False,
        width=10000,
        height=len(df.index) * 30,  # Adjusting height based on number of characters
        margin=dict(t=50, r=50, b=100, l=200),  # Adjust margins to fit labels
        yaxis=dict(tickangle=-30),  # Adjust y-axis tick angle for better readability
    )
    output_directory = "output_onstage_heatmap"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Get the output filename based on the mapping
    output_file_name = get_output_filename(os.path.basename(file_path))
    output_file_path = os.path.join(output_directory, output_file_name)
    fig.write_html(output_file_path)

# List all .xml files in the 'Data' directory
xml_files = [os.path.join("data", f) for f in os.listdir("data") if f.endswith('.xml')]

# Process each .xml file
for file in xml_files:
    process_xml_to_heatmap(file)