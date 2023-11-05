import os
import json
from bs4 import BeautifulSoup
import pandas as pd
import re

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the "data" directory and "output" directory
data_dir = os.path.join(script_directory, "data")
output_dir = os.path.join(script_directory, "output")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def count_words(tag):
    return re.findall(r'\S+', tag.get_text())

def extract_text_from_ab(tag):
    words_list = [w_tag.get_text() for w_tag in tag.find_all('w')]
    return " ".join(words_list)

files = os.listdir(data_dir)
for file_name in files:
    if not file_name.endswith('.xml'):
        continue

    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    soup = BeautifulSoup(content, 'lxml-xml')  
    on_stage = []
    exchanges = []

    for tag in soup.find_all(['sp', 'stage']):
        if tag.name == 'sp' and 'who' in tag.attrs:
            speaker = tag['who'].replace("#", "")
            others = [char for char in on_stage if char != speaker]
            if not others:
                others.append(speaker)

            ab_tag = tag.find('ab')
            if ab_tag:  # Ensure that an <ab> tag exists
                words_spoken = extract_text_from_ab(ab_tag)
                word_count = len(re.findall(r'\S+', words_spoken))

                for listener in others:
                    exchanges.append({
                        'speaker': speaker,
                        'listener': listener,
                        'words': word_count,
                        'text': words_spoken
                    })
             
        if tag.name == 'stage' and 'who' in tag.attrs:
            if tag.get('type') == 'entrance':
                characters_entering = tag['who'].replace("#", "").split()
                on_stage = list(set(on_stage + characters_entering))
            elif tag.get('type') == 'exit':
                characters_exiting = tag['who'].replace("#", "").split()
                on_stage = [char for char in on_stage if char not in characters_exiting]

    df = pd.DataFrame(exchanges)
    df_summary = df.groupby(['speaker', 'listener']).agg(total_words=('words', 'sum')).reset_index()

    # Save results as JSON
    output_filename = file_name.replace(".xml", ".json")
    output_path = os.path.join(output_dir, output_filename)
    
    results = {
        "individual_exchanges": exchanges,
        "word_counts_summary": df_summary.to_dict(orient="records")
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Processed {file_name} and saved results to {output_filename}")
