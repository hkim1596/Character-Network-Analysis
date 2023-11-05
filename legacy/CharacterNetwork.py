from bs4 import BeautifulSoup
import pandas as pd
import re

# Open and read the XML file
with open('data/Ham.xml', 'r', encoding='utf-8') as file:
    content = file.read()

# Parse the XML content using BeautifulSoup
soup = BeautifulSoup(content, 'lxml-xml')  # Note the 'lxml-xml' parser

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

# Print result
print(word_counts)

# Save to a CSV
word_counts.to_csv('word_counts_output.csv', index=False)
