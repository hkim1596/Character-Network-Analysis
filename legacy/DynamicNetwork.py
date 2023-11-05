from bs4 import BeautifulSoup
import pandas as pd
import re
from bokeh.io import output_file, show, push_notebook
from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges
from bokeh.layouts import column
from bokeh.models.widgets import Slider
from bokeh.layouts import layout
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
import networkx as nx
from bokeh.plotting import figure, from_networkx
from bokeh.layouts import column
from bokeh.models import Slider
from bokeh.plotting import figure, from_networkx, output_file, show



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

# List to store dynamic network states
network_states = []

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
    
    current_state = word_counts.copy()
    network_states.append(current_state)

# Summarize word counts
word_counts = word_counts.groupby(['speaker', 'listener']).agg(total_words=('words', 'sum')).reset_index()

# Print result
print(word_counts)

# Save to a CSV
word_counts.to_csv('word_counts_output.csv', index=False)

# Visualization using Bokeh

output_file("dynamic_network.html")

def get_plot_for_state(state_num):
    G = nx.from_pandas_edgelist(network_states[state_num], 'speaker', 'listener', 'words')
    plot = figure(width=800, height=800, x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    graph_renderer = from_networkx(G, nx.spring_layout, scale=1, center=(0,0))
    plot.renderers.append(graph_renderer)
    
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color='blue')
    graph_renderer.edge_renderer.glyph = MultiLine(line_color="green", line_alpha=0.8, line_width=1)
    
    return plot

def update(attr, old, new):
    layout.children[1] = get_plot_for_state(slider.value)

initial_plot = get_plot_for_state(0)
slider = Slider(start=0, end=len(network_states)-1, value=0, step=1, title="Step")
slider.on_change('value', update)

layout = column(slider, initial_plot)
show(layout)