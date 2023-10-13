import os
from bs4 import BeautifulSoup
import pandas as pd
import re

# Directory containing XML files
data_dir = "data"
output_dir = "output"

# List all files in the directory
files = os.listdir(data_dir)

# If the output directory doesn't exist, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each file
for file_name in files:
    # Check if the file is an XML file (optional)
    if not file_name.endswith('.xml'):
        continue

    file_path = os.path.join(data_dir, file_name)

    # Open and read the XML file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Parse the XML content using BeautifulSoup
    soup = BeautifulSoup(content, 'lxml-xml')  

    # Initialize an empty list to track characters on stage
    on_stage = []

    # Initialize a dataframe for results
    word_counts = pd.DataFrame(columns=['speaker', 'listener', 'words'])

    # Count words in a tag
    def count_words(tag):
        return len(re.findall(r'\S+', tag.get_text()))

    # Iterate through all nodes
    for tag in soup.find_all(['sp', 'stage']):
        # Check if the tag is a speech
        if tag.name == 'sp' and 'who' in tag.attrs:
            speaker = tag['who'].replace("#", "")
            others = [char for char in on_stage if char != speaker] 
            
            # If the speaker is alone on stage, add them to the 'others' list to count words as speaking to themselves
            if not others:
                others.append(speaker)
            
            word_count = count_words(tag)
            
            for listener in others:
                new_row = pd.DataFrame({'speaker': [speaker], 'listener': [listener], 'words': [word_count]})
                word_counts = pd.concat([word_counts, new_row], ignore_index=True)
                
        # Check if the tag is a stage direction for entrance
        if tag.name == 'stage' and tag.get('type') == 'entrance' and 'who' in tag.attrs:
            characters_entering = tag['who'].replace("#", "").split()
            on_stage = list(set(on_stage + characters_entering))

        # Check if the tag is a stage direction for exit
        if tag.name == 'stage' and tag.get('type') == 'exit' and 'who' in tag.attrs:
            characters_exiting = tag['who'].replace("#", "").split()
            on_stage = [char for char in on_stage if char not in characters_exiting]

    # Summarize word counts
    word_counts = word_counts.groupby(['speaker', 'listener']).agg(total_words=('words', 'sum')).reset_index()


    # Save to a CSV file with a matching name
    output_filename = file_name.replace(".xml", ".csv")
    output_path = os.path.join(output_dir, output_filename)
    word_counts.to_csv(output_path, index=False)

    print(f"Processed {file_name} and saved results to {output_filename}")

