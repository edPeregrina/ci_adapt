####
##WORK IN PROGRESS, USE WITH CAUTION
### 
""""This commit is able to create a network with 
OSM rail data. The rail network is created using 
snkit and simplified using work from Asgarpour. 

The network flow model uses the concept approach
proposed by Meijer for water flow which has been tested
for road networks in <Hackathon> demonstrating 
flexibility of application to multiple types of 
networks. 

The graph is solved using the min cost flow algorithm 
from networkx. The flow is then visualized using networkx. 

The next steps are to add costs to the edges for 
economic appraisal.
Demand data: EUSTAT rail_tf_ns20_de
"""	
#ADDITIONAL DEPENDENCIES
# #!pip install triangle
# #!pip install rasterio
# #!pip install geopy
# #!pip install osmnx

#IMPORTS
import sys
sys.path.append(r'C:\repos\snkit\src')
sys.path.append(r'C:\repos\ra2ce')
sys.path.append(r'C:\repos\ra2ce_multi_network')
from ci_adapt_utilities import *
from ci_adapt_classes import *
from ra2ce_multi_network.simplify_rail import *
from ra2ce_multi_network.simplify_rail import _merge_edges, _network_to_nx
import networkx as nx
import pandas as pd
from shapely import wkt, box

#FUNCTION DEFINITIONS
#TODO: Add cost for economic appraisal, add mode factor to edge weights for multi-modality
def get_bbox(graph):
    min_x = min([attr['pos'][0] for _, attr in graph.nodes(data=True)])
    max_x = max([attr['pos'][0] for _, attr in graph.nodes(data=True)])
    min_y = min([attr['pos'][1] for _, attr in graph.nodes(data=True)])
    max_y = max([attr['pos'][1] for _, attr in graph.nodes(data=True)])
    return min_x, max_x, min_y, max_y

def set_edge_weights(assets, graph):
    asset_lengths = {str(asset['osm_id']): asset['geometry'].length for asset_id, asset in assets.iterrows()}
    for u, v, attr in graph.edges(data=True):
        if 'source_sink' in str(u) or 'source_sink' in str(v):
            continue
        attr['weight'] = int(0)
        attr['length'] = 0
        if 'osm_id' not in attr:
            continue
        osm_ids = attr['osm_id'].split('; ')
        for osm_id in osm_ids:
            if osm_id in asset_lengths:
                attr['length'] += asset_lengths[osm_id]
                attr['weight'] += int(round(asset_lengths[osm_id]*1e3,0))

    return graph

def set_edge_capacities(graph):
    for _, _, _, attr in graph.edges(keys=True, data=True):
        if 'maxspeed' not in attr:
            attr['capacity'] = int(100000)
            continue
        asset_speeds = attr['maxspeed'].split('; ') if attr['maxspeed'] else []
        if len(asset_speeds) > 0:
            attr['capacity'] = 1000*int(min(asset_speeds)) #TODO: Add capacity based on speed? area average trains per year?
        else: attr['capacity'] = int(100)
    return graph

def nearest_nodes(graph, point, n):
    nearest_nodes = []
    for node, attr in graph.nodes(data=True):
        distance = point.distance(attr['geometry'])
        nearest_nodes.append((node, distance))
    nearest_nodes = sorted(nearest_nodes, key=lambda x: x[1])
    return nearest_nodes[:n]

def set_node_demand(od_data, graph):
    for od in od_data.iterrows():
        if od[1]['geometry'].geom_type == 'Point':
            centroid = od[1]['geometry']
        else:
            centroid = od[1]['geometry'].centroid
        
        nearest_node = nearest_nodes(graph, centroid, 1)
        graph.nodes[nearest_node[0][0]]['demand'] += int(od[1]['demand']-od[1]['supply'])
    return graph

def equalize_demand_supply(graph):
    sumdem=0 
    for u,attr in graph.nodes(data=True):
        if attr['demand'] !=0:
            sumdem+=int(attr['demand'])

    graph.add_node('source_sink', demand=int(-sumdem), pos=graph.nodes[0]['pos'])
    for u, attr in graph.nodes(data=True):
        if u == 'source_sink':
            continue
        if not graph.has_edge(u, 'source_sink', 0):
            graph.add_edge(u,'source_sink', 0, weight=int(1e8))
        if not graph.has_edge('source_sink', u):
            graph.add_edge('source_sink', u, 0, weight=int(1e8))

    return graph

def inspect_graph(graph):
    edge_capacities_types = []
    edge_weights_types = []
    node_demands_types = []

    for _, _, attr in graph.edges(data=True):
        if 'capacity' in attr:
            edge_capacities_types.append(type(attr['capacity']))
        if 'weight' in attr:
            edge_weights_types.append(type(attr['weight']))

    for _, attr in graph.nodes(data=True):
        if 'demand' in attr:
            node_demands_types.append(type(attr['demand']))

    return edge_capacities_types, edge_weights_types, node_demands_types

#MAIN
# configure handler with ini file. ini file can be created with config.py
H=Handler(config_file='config_ci_adapt.ini')

# read exposure data
H.read_asset_data()

#Source: Asgarpour/snkit
# Create a railway networks with possible terminal nodes. This returns a complex network, as includes the rail tracks with the highest level of detail.
aggregation_range = 0.08 # in km
complex_rail_network = get_rail_network_with_terminals(network_gdf=H.assets.assets, aggregation_range=aggregation_range)
merged_rail_network = _merge_edges(network=complex_rail_network, excluded_edge_types=['bridge', 'tunnel']) #Must add network= to pass excluded_edge_types as a keyword argument

# Number of nodes and edges reduced
print(f"Difference in node counts: {complex_rail_network.nodes.shape[0] - merged_rail_network.nodes.shape[0]}")
print(f"Difference in node counts %: {round(100*(complex_rail_network.nodes.shape[0] - merged_rail_network.nodes.shape[0])/complex_rail_network.nodes.shape[0], 0)}")
print(f"Difference in edge counts: {complex_rail_network.edges.shape[0] - merged_rail_network.edges.shape[0]}")
print(f"Difference in edge counts %: {round(100*(complex_rail_network.edges.shape[0] - merged_rail_network.edges.shape[0])/complex_rail_network.edges.shape[0], 0)}")

merged_rail_graph = _network_to_nx(merged_rail_network)

G=nx.MultiDiGraph(merged_rail_graph)

for _, attr in G.nodes(data=True):
    attr['demand'] = int(0)
    attr['pos'] = (attr['geometry'].x, attr['geometry'].y)

bbox=get_bbox(G)

od_data_source = r'C:\Data\interim\od_data_2.csv'
od_data = pd.read_csv(od_data_source)
# Drop rows where 'geometry' is NaN
od_data = od_data.dropna(subset=['geometry'])
od_data['geometry'] = od_data['geometry'].apply(wkt.loads)
bbox_shape=box(bbox[0],bbox[1], bbox[2], bbox[3])
od_data = od_data[od_data['geometry'].apply(lambda geom: bbox_shape.contains(geom))]

G = set_node_demand(od_data, G)
G = equalize_demand_supply(G)
G = set_edge_capacities(G)
G = set_edge_weights(H.assets.assets, G)

#Source: SA
max_weight_graph = max(attr['weight'] for _, _, attr in G.edges(data=True))
print('Max weight: '+str(max_weight_graph))
max_capacity_graph = max(attr['capacity'] for _, _, attr in G.edges(data=True))
print('Max capacity: '+str(max_capacity_graph))

# create a virtual node with dummy nodes
G_v=G.copy()
# convert to int
for u, v, key, attr in G.edges(keys=True, data=True):
    G_v.add_edge((str(u) + '_d'), (str(v) + '_d'), **attr)

for u in G.nodes:
    G_v.add_edge(u,(str(u) + '_d'),capacity=max_capacity_graph*100,weight=int(round(max_weight_graph*1e3,0)))
    G_v.add_edge((str(u) + '_d'),u,capacity=max_capacity_graph*100,weight=0)

# verify capacities, weights and demands are integers
edge_capacities_types, edge_weights_types, node_demands_types = inspect_graph(G_v)
print("Edge Capacities Types:", set(edge_capacities_types))
print("Edge Weights Types:", set(edge_weights_types))
print("Node Demands Types:", set(node_demands_types))
if {type(int())} == set(list(edge_capacities_types) + list(edge_weights_types) + list(node_demands_types)):
    print('Success: only int type values')
else: 
    print('Warning! Not all values are integers')

# solve the min cost flow problem
mcf=nx.min_cost_flow(G_v)
print('Minimum cost flow: '+str(mcf))

flow_dict = mcf

#visualisation
import re
from math import ceil
extract_numeric_part = lambda s: int(re.search(r'\d+', s).group()) if re.search(r'\d+', s) else None

for u,attr in G_v.nodes(data=True):
    try:
        if isinstance(u, int): 
            pass
        elif isinstance(u, str): 
            if u == 'source_sink_d':
                attr['pos']=(G.nodes['source_sink']['pos'][0],G_v.nodes['source_sink']['pos'][-1])
            else:
                attr['pos']=(G_v.nodes[extract_numeric_part(u)]['pos'][0]+0.0005,G_v.nodes[extract_numeric_part(u)]['pos'][-1]+0.0005)
    except:
        print(f'Error in setting node {u} position')    
pos=nx.get_node_attributes(G_v, 'pos')
nodelist=[u for u, attr in G_v.nodes(data=True) if 'demand' in attr and attr['demand'] != 0]
edgelist=[(u, v) for u, v, attr in G.edges(data=True) if (type(u) == int and type(v) == int)]
node_size=[int(abs(ceil(attr['demand']/100))) for u, attr in G_v.nodes(data=True) if 'demand' in attr and attr['demand'] != 0]
width=[int(ceil(mcf[u][v][0]/500)+1) for u, v in edgelist]
# draw network
nx.draw_networkx(G_v, pos,
        nodelist=nodelist,
        edgelist=edgelist,
        with_labels=False, 
        arrows=False,
        node_size=node_size,
        node_color=['red' if attr['demand'] > 0 else 'blue' for u, attr in G_v.nodes(data=True) if 'demand' in attr and attr['demand'] != 0],
        width=width,
        edge_color=None)


