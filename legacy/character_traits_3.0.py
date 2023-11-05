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

# def clean_character_name(name, suffix):
#     """
#     Cleans up the character name by:
#     - Removing the provided suffix.
#     - Capitalizing the first character of each word.
#     - Replacing any punctuation with a space.
#     """
#     name = name.replace(suffix, '')  # Remove suffix
#     name = ' '.join(word.capitalize() for word in name.split('_'))  # Capitalize and replace underscores with spaces
#     name = re.sub(r'[^\w\s]', ' ', name)  # Replace punctuation with space
#     return name.title().strip()

# Process each file
for file_name in files:
    # Check if the file is an XML file
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

    # Count words in a tag
    def count_words(tag):
        return len(re.findall(r'\S+', tag.get_text()))

    interactions = {}

    # Iterate through all nodes
    for tag in soup.find_all(['sp', 'stage']):
         # Stage direction for entrance
        if tag.name == 'stage' and tag.get('type') == 'entrance' and 'who' in tag.attrs:
            characters_entering = [char for char in tag['who'].replace("#", "").split()]
            on_stage = list(set(on_stage + characters_entering))

        # Stage direction for exit
        if tag.name == 'stage' and tag.get('type') == 'exit' and 'who' in tag.attrs:
            characters_exiting = [char for char in tag['who'].replace("#", "").split()]
            on_stage = [char for char in on_stage if char not in characters_exiting]

       # Check if the tag is a speech
        if tag.name == 'sp' and 'who' in tag.attrs:
            speaker = tag['who'].replace("#", "")
            others = [char for char in on_stage if char != speaker]
            
            if not others:
                others.append(speaker)
            
            word_count = count_words(tag)
            
            for listener in others:
                interactions.setdefault(speaker, {}).setdefault(listener, 0)
                interactions[speaker][listener] += word_count


    # Convert interactions dictionary to dataframe
    matrix_df = pd.DataFrame(interactions).fillna(0)

    # Clean character names
    suffix_to_remove = f"_{file_name.replace('.xml', '')}"
    matrix_df.columns = [clean_character_name(name, suffix_to_remove) for name in matrix_df.columns]
    matrix_df.index = [clean_character_name(name, suffix_to_remove) for name in matrix_df.index]

    # Make sure the matrix is square
    all_characters = list(set(matrix_df.index).union(set(matrix_df.columns)))
    matrix_df = matrix_df.reindex(index=all_characters, columns=all_characters).fillna(0)

    # Calculate total words spoken by each character
    total_words = matrix_df.sum(axis=1)

    # Sort characters by total words spoken in descending order
    sorted_characters = total_words.sort_values(ascending=False).index.tolist()

    # Rearrange the DataFrame rows and columns based on the sorted order
    matrix_df = matrix_df.loc[sorted_characters, sorted_characters]

    # Save to a CSV file with a matching name
    output_filename = file_name.replace(".xml", "_matrix.csv")
    output_path = os.path.join(output_dir, output_filename)
    matrix_df.to_csv(output_path, index=True)

    print(f"Processed {file_name} and saved results to {output_filename}.")
    print(f"The generated CSV file has {matrix_df.shape[0]} rows and {matrix_df.shape[1]} columns.")

