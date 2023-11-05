import os
from bs4 import BeautifulSoup
import pandas as pd
import re

data_dir = "data"
output_dir = "output_matrix"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def clean_character_name(name, suffix):
    name = name.replace(suffix, '')
    name = ' '.join(word.capitalize() for word in name.split('_'))
    name = re.sub(r'[^\w\s]', ' ', name)
    return name.title().strip()

# Count words in a tag
def count_words(tag):
    ab_tag = tag.find('ab')
    if ab_tag:
        return len(re.findall(r'\S+', ab_tag.get_text()))
    return 0

def process_xml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Parse the XML content using BeautifulSoup
    soup = BeautifulSoup(content, 'lxml-xml')
    
    # Initialize an empty list to track characters on stage
    on_stage = []
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
            speakers = tag['who'].replace("#", "").split()  # Split speakers into a list
            for speaker in speakers:  # Loop through each speaker
                others = [char for char in on_stage if char != speaker]  # Listeners for this specific speaker
                if not others:
                    others.append(speaker)
                
                word_count = count_words(tag) // len(speakers)  # Divide the word count among the speakers
                
                for listener in others:
                    interactions.setdefault(speaker, {}).setdefault(listener, 0)
                    interactions[speaker][listener] += word_count

    return interactions

for file_name in [f for f in os.listdir(data_dir) if f.endswith('.xml')]:
    interactions = process_xml(os.path.join(data_dir, file_name))
    
    # Convert interactions dictionary to dataframe and transpose
    matrix_df = pd.DataFrame(interactions).T.fillna(0)

    all_characters = list(set(matrix_df.index).union(set(matrix_df.columns)))
    matrix_df = matrix_df.reindex(index=all_characters, columns=all_characters).fillna(0)
    total_words = matrix_df.sum(axis=1)
    sorted_characters = total_words.sort_values(ascending=False).index.tolist()
    matrix_df = matrix_df.loc[sorted_characters, sorted_characters]

    suffix_to_remove = f"_{file_name.replace('.xml', '')}"
    matrix_df.columns = [clean_character_name(name, suffix_to_remove) for name in matrix_df.columns]
    matrix_df.index = [clean_character_name(name, suffix_to_remove) for name in matrix_df.index]
    
    # Save to a CSV file with a matching name
    output_filename = file_name.replace(".xml", "_matrix.csv")
    output_path = os.path.join(output_dir, output_filename)
    matrix_df.to_csv(output_path, index=True)
    print(f"Processed {file_name} and saved results to {output_filename}.")
    print(f"The generated CSV file has {matrix_df.shape[0]} rows and {matrix_df.shape[1]} columns.")
