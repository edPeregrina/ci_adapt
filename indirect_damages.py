####
##WORK IN PROGRESS, USE WITH CAUTION
###
#TODO make node and edge ids (keys) strings instead of integers
import pickle
import numpy as np
# from shapely.validation import make_valid # only needed to make invalid geometries valid
from ci_adapt_utilities import *
from ci_adapt_classes import *

# configure handler with ini file. ini file can be created with config.py
H=Handler(config_file='config_ci_adapt.ini')
# read vulnerability and maximum damage data
H.read_vul_maxdam()

# read hazard data
H.read_hazard_data()

# read exposure data
H.read_asset_data()

# #!pip install triangle
# #!pip install rasterio
# #!pip install geopy
# #!pip install osmnx
import sys
# import os
sys.path.append(r'C:\repos\snkit\src')
sys.path.append(r'C:\repos\ra2ce')
sys.path.append(r'C:\repos\ra2ce_multi_network')
# from ra2ce_multi_network.deeper_extraction import filter_on_other_tags
from ra2ce_multi_network.simplify_rail import *

# Create a railway networks with possible terminal nodes. This returns a complex network, as includes the rail tracks with the highest level of detail.
aggregation_range = 0.08 # in km
complex_rail_network = get_rail_network_with_terminals(network_gdf=H.assets.assets, aggregation_range=aggregation_range)


# making merged rail network #before this: create snkit network
from ra2ce_multi_network.simplify_rail import _merge_edges, _network_to_nx
import networkx as nx
merged_rail_network = _merge_edges(network=complex_rail_network, excluded_edge_types=['bridge', 'tunnel']) #Must add network= to pass excluded_edge_types as a keyword argument

# Number of nodes and edges redduced

print(f"Difference in node counts: {complex_rail_network.nodes.shape[0] - merged_rail_network.nodes.shape[0]}")
print(f"Difference in node counts %: {round(100*(complex_rail_network.nodes.shape[0] - merged_rail_network.nodes.shape[0])/complex_rail_network.nodes.shape[0], 0)}")

print(f"Difference in edge counts: {complex_rail_network.edges.shape[0] - merged_rail_network.edges.shape[0]}")
print(f"Difference in edge counts %: {round(100*(complex_rail_network.edges.shape[0] - merged_rail_network.edges.shape[0])/complex_rail_network.edges.shape[0], 0)}")


merged_rail_graph = _network_to_nx(merged_rail_network)

G=nx.MultiDiGraph(merged_rail_graph)

for _, _, attr in G.edges(data=True):
    attr['weight']=1
    attr['capacity']=10

for _, attr in G.nodes(data=True):
    attr['demand']=0
    attr['pos'] = (attr['geometry'].x, attr['geometry'].y)



#SA
max_weight_graph = max(attr['weight'] for _, _, attr in G.edges(data=True))
print('Max weight: '+str(max_weight_graph))
max_capacity_graph = max(attr['capacity'] for _, _, attr in G.edges(data=True))
print('Max capacity: '+str(max_capacity_graph))

G_v=G.copy()
#convert to int
for u, v, key, attr in G.edges(keys=True, data=True):
    attr['weight'] = int(round(attr['weight'] * 10e3, 0))
    attr['capacity'] = int(max_capacity_graph * 100)
    G_v.add_edge((str(u) + '_d'), (str(v) + '_d'), **attr)

for u in G.nodes:
    G_v.add_edge(u,(str(u) + '_d'),capacity=max_capacity_graph*100,weight=max_weight_graph*1e3)
    G_v.add_edge((str(u) + '_d'),u,capacity=max_capacity_graph*100,weight=0)


mcf=nx.min_cost_flow(G_v)
print('Minimum cost flow: '+str(mcf))

flow_dict = mcf


import re

extract_numeric_part = lambda s: int(re.search(r'\d+', s).group()) if re.search(r'\d+', s) else None


for u,attr in G_v.nodes(data=True):
    if isinstance(u, int): 
        pass
    elif isinstance(u, str): 
        attr['pos']=(G_v.nodes[extract_numeric_part(u)]['pos'][0]+0.0005,G_v.nodes[extract_numeric_part(u)]['pos'][-1]+0.0005)
    
pos=nx.get_node_attributes(G_v, 'pos')

draw_edges=nx.draw_networkx_edges(G_v,pos)
nx.draw(G_v, pos, 
        with_labels=False,
        node_size=0,
        node_color='grey')
