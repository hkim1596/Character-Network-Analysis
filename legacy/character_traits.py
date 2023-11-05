import os
from bs4 import BeautifulSoup
import pandas as pd
import re

# Directory containing XML files
data_dir = "data"
output_dir = "output_matrix"

# List all files in the directory
files = os.listdir(data_dir)

# If the output directory doesn't exist, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def strip_suffix_from_name(name, suffix):
    # Function to remove the given suffix from the character name
    return name.replace(suffix, "")

def strip_suffix_from_name(name, suffix):
    # Function to remove the given suffix from the character name
    name = name.replace(suffix, "")
    # Capitalize the first letter of each word
    return name.title()

# Process each file
for file_name in files:
    # Check if the file is an XML file (optional)
    if not file_name.endswith('.xml'):
        continue

    file_path = os.path.join(data_dir, file_name)

    # Determine suffix based on filename
    suffix = f"_{file_name.replace('.xml', '')}"

    # Open and read the XML file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Parse the XML content using BeautifulSoup
    soup = BeautifulSoup(content, 'lxml-xml')  

    # Initialize an empty list to track characters on stage
    on_stage = []

    # Count words in a tag
    def count_words(tag):
        return len(re.findall(r'\S+', tag.get_text()))

    interactions = {}

    # Iterate through all nodes
    for tag in soup.find_all(['sp', 'stage']):
        # Check if the tag is a speech
        if tag.name == 'sp' and 'who' in tag.attrs:
            speaker = strip_suffix_from_name(tag['who'].replace("#", ""), suffix)
            others = [strip_suffix_from_name(char, suffix) for char in on_stage if char != speaker]
            
            if not others:
                others.append(speaker)
            
            word_count = count_words(tag)
            
            for listener in others:
                interactions.setdefault(speaker, {}).setdefault(listener, 0)
                interactions[speaker][listener] += word_count

        # Stage direction for entrance
        if tag.name == 'stage' and tag.get('type') == 'entrance' and 'who' in tag.attrs:
            characters_entering = [strip_suffix_from_name(char, suffix) for char in tag['who'].replace("#", "").split()]
            on_stage = list(set(on_stage + characters_entering))

        # Stage direction for exit
        if tag.name == 'stage' and tag.get('type') == 'exit' and 'who' in tag.attrs:
            characters_exiting = [strip_suffix_from_name(char, suffix) for char in tag['who'].replace("#", "").split()]
            on_stage = [char for char in on_stage if char not in characters_exiting]

    # Convert interactions dictionary to dataframe
    matrix_df = pd.DataFrame(interactions).fillna(0).T.fillna(0)

    # Calculate total words spoken by each character
    total_words = matrix_df.sum(axis=1)

    # Sort the dataframe based on the total words spoken
    sorted_characters = total_words.sort_values(ascending=False).index
    matrix_df = matrix_df.reindex(sorted_characters, axis=0).reindex(sorted_characters, axis=1)

    # Save to a CSV file with a matching name
    output_filename = file_name.replace(".xml", "_matrix.csv")
    output_path = os.path.join(output_dir, output_filename)
    matrix_df.to_csv(output_path, index=True)

    print(f"Processed {file_name} and saved results to {output_filename}")