import numpy as np
import pandas as pd
import pickle
import geopandas as gpd
from tqdm import tqdm
import datetime
import shapely
from shapely import Point, box, length, intersects, intersection, make_valid, is_valid
from direct_damages import damagescanner_rail_track as ds
import re
from pyproj import Transformer
from math import ceil
import networkx as nx
from pathlib import Path
import pathlib
import configparser



def preprocess_assets(assets_path):
    """
    Preprocesses asset data by reading from a file, creating a GeoDataFrame, reprojecting it, filtering for railway freight line assets, and renaming columns.

    Args:
        assets_path (str): Path to the asset data file.

    Returns:
        GeoDataFrame: Preprocessed asset data.
    """

    assets = gpd.read_file(assets_path)
    assets = gpd.GeoDataFrame(assets).set_crs(4326).to_crs(3857)
    assets = assets.loc[assets.geometry.geom_type == 'LineString']
    assets = assets.rename(columns={'railway' : 'asset'})
    # assets = assets[assets['railway:traffic_mode']!='"passenger"']
    assets = assets[assets['asset']=='rail']

    assets = assets.reset_index(drop=True)
    
    return assets

def process_asset_options(asset_options, map_rp_spec, rp_spec_priority):
    """
    Determines whether to skip processing bridges and tunnels based on their design return periods compared to the map return period.

    Args:
        asset_options (dict): Dictionary of asset options.
        map_rp_spec (str): Map return period specification.
        rp_spec_priority (list): List of return period priorities.

    Returns:
        tuple: Boolean values indicating whether to skip bridges and tunnels.
    """    
    map_rp_spec_index=rp_spec_priority.index(map_rp_spec)  
    if 'bridge_design_rp' in asset_options.keys():
        bridge_design_rp = asset_options['bridge_design_rp']
        bridge_design_rp_index=rp_spec_priority.index(bridge_design_rp)
        if bridge_design_rp_index <= map_rp_spec_index:
            skip_bridge=True
        else:
            skip_bridge=False
    if 'tunnel_design_rp' in asset_options.keys():
        tunnel_design_rp = asset_options['tunnel_design_rp']
        tunnel_design_rp_index=rp_spec_priority.index(tunnel_design_rp)
        if tunnel_design_rp_index <= map_rp_spec_index:
            skip_tunnel=True
        else:
            skip_tunnel=False

    return skip_bridge, skip_tunnel

def get_number_of_lines(asset):
    """
    Extracts the number of 'passenger lines' from the asset's 'other_tags' field. Note these are not necessarily passenger lines, but rather the number of tracks.

    Args:
        asset (Series): Asset data, an element of the assets GeoDataFrame.

    Returns:
        int: Number of passenger lines.
    """
    asset_other_tags = asset['other_tags']
    if asset_other_tags is None:
        number_of_lines = 1
        return number_of_lines
    search = re.search('passenger_lines', asset_other_tags)
    if search:
        group_end = search.span()[-1]
        number_of_lines=asset_other_tags[group_end:].split('"=>"')[1].split('"')[0]    
    else:
        number_of_lines = 1
    
    return number_of_lines

def process_hazard_data(single_footprint, hazard_type, assets, interim_data_path, infra_curves, max_damage_tables, curve_types, infra_type, type_dict, geom_dict, asset_options=None, rp_spec_priority = None):
    """
    Processes hazard data, overlays it with assets, and calculates potential damages using infrastructure curves and maximum damage tables.

    Args:
        single_footprint (Path): Path to the hazard footprint file.
        hazard_type (str): Type of hazard.
        assets (GeoDataFrame): Asset data.
        interim_data_path (Path): Path to interim data storage.
        infra_curves (dict): Infrastructure damage curves from vulnerability data.
        max_damage_tables (DataFrame): Maximum damage tables with the replacement cost of different assets (complements infra_curves).
        curve_types (dict): Curve asset types and their corresponding curve IDs, e.g. {'rail': ['F8.1']}
        infra_type (str): Infrastructure type, e.g. 'rail'.
        type_dict (dict): Dictionary of asset types.
        geom_dict (dict): Dictionary of asset geometries.
        asset_options (dict, optional): Dictionary of asset options. Defaults to None.
        rp_spec_priority (list, optional): List of return period priorities. Defaults to None.

    Returns:
        dict: Dictionary of damages per asset.
    """
    hazard_name = single_footprint.parts[-1].split('.')[0]
    map_rp_spec = hazard_name.split('_')[3]
    if asset_options is not None and rp_spec_priority is not None:
        skip_bridge, skip_tunnel = process_asset_options(asset_options, map_rp_spec, rp_spec_priority)
    else:
        skip_bridge=False
        skip_tunnel=False

    # load hazard map
    if hazard_type in ['pluvial','fluvial']:
        hazard_map = ds.read_flood_map(single_footprint)
    else: 
        print(f'{hazard_type} not implemented yet')
        return 

    # convert hazard data to epsg 3857
    if '.shp' or '.geojson' in str(hazard_map):
        hazard_map=gpd.read_file(hazard_map).to_crs(3857)[['w_depth_l','w_depth_u','geometry']] #take only necessary columns (lower and upper bounds of water depth and geometry)
    else:
        hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'{timestamp} - Coarse overlay of hazard map with assets...')
    
    # make geometry valid
    hazard_map['geometry'] = hazard_map['geometry'].make_valid() if not hazard_map['geometry'].is_valid.all() else hazard_map['geometry']

    # coarse overlay of hazard map with assets
    intersected_assets=ds.overlay_hazard_assets(hazard_map,assets)
    overlay_assets = pd.DataFrame(intersected_assets.T,columns=['asset','hazard_point'])

    # convert dataframe to numpy array
    # considering upper and lower bounds #TODO improve logic, counterintuitive
    hazard_numpified_l = hazard_map.drop('w_depth_u', axis=1).to_numpy() # lower bound, dropping upper bound data
    hazard_numpified_u = hazard_map.drop('w_depth_l', axis=1).to_numpy() # upper bound, dropping lower bound data
    hazard_numpified_list=[hazard_numpified_l, hazard_numpified_u] 

    # pickle asset overlays and hazard numpified data for use in adaptation
    overlay_path = f'{interim_data_path}/overlay_assets_{hazard_name}.pkl'

    with open(overlay_path, 'wb') as f:
        pickle.dump(overlay_assets, f)
    hazard_numpified_path = f'{interim_data_path}/hazard_numpified_{hazard_name}.pkl'    
    with open(hazard_numpified_path, 'wb') as f:
        pickle.dump(hazard_numpified_list, f)  

    # iterate over the infrastructure curves and collect in-between results
    for infra_curve in infra_curves:
        maxdams_filt=max_damage_tables[max_damage_tables['ID number']==infra_curve[0]]['Amount'] # can also be made neater
        if not infra_curve[0] in curve_types[infra_type]:
            continue
        
        # get curves
        curve = infra_curves[infra_curve[0]]
        hazard_intensity = curve.index.values
        fragility_values = (np.nan_to_num(curve.values,nan=(np.nanmax(curve.values)))).flatten()       

        # dictionary of unique assets and their damage (one per map)
        collect_inb = {}

        for asset in tqdm(overlay_assets.groupby('asset'),total=len(overlay_assets.asset.unique())): #group asset items for different hazard points per asset and get total number of unique assets
            # verify asset has an associated asset type (issues when trying to drop bridges, dictionaries have to reset)
            try:
                asset_type = type_dict[asset[0]]
            except KeyError: 
                print(f'Passed asset! {asset[0]}')
                continue
            
            # check if the asset type has a matching vulnerability curve
            if not infra_curve[0] in curve_types[asset_type]: 
                collect_inb[asset[0]] = 0
                print(f'Asset {asset[0]}: No vulnerability data found')

            # check if there are non-0 fragility values
            if np.max(fragility_values) == 0:
                collect_inb[asset[0]] = 0  
                print(f'Asset {asset[0]}: Fragility = 0')
            else:
                if assets.loc[asset[0]].bridge == 'yes' and skip_bridge==True:
                    collect_inb[asset[0]] = (0, 0)
                    continue
                if assets.loc[asset[0]].tunnel == 'yes' and skip_tunnel==True:
                    collect_inb[asset[0]] = (0, 0)
                    continue
                number_of_lines = get_number_of_lines(assets.loc[asset[0]])
                if int(number_of_lines) == 2:
                    double_track_factor = 0.5
                else: 
                    double_track_factor = 1.0
                # retrieve asset geometry and do fine overlay
                asset_geom = geom_dict[asset[0]]              
                # get damage per asset in a single hazard map as a dictionary of asset IDs:damage tuples
                collect_inb[asset[0]] = tuple(ds.get_damage_per_asset(asset,h_numpified,asset_geom,hazard_intensity,fragility_values,maxdams_filt, double_track_factor)[0] for h_numpified in hazard_numpified_list)

    return collect_inb

def retrieve_max_intensity_by_asset(asset, overlay_assets, hazard_numpified_list):
    """
    Retrieves the maximum hazard intensity intersecting with a specific asset. The upper bound is used.

    Args:
        asset (str): Asset identifier.
        overlay_assets (DataFrame): DataFrame of overlay assets.
        hazard_numpified_list (list): List of hazard intensity arrays.

    Returns:
        ndarray: Maximum hazard intensity values.
    """
    max_intensity = hazard_numpified_list[-1][overlay_assets.loc[overlay_assets['asset'] == asset].hazard_point.values] 
    return max_intensity[:,0]

def set_rp_priorities(return_period_dict):
    """
    Orders return periods from highest to lowest priority.

    Args:
        return_period_dict (dict): Dictionary of return periods.

    Returns:
        tuple: Ordered return periods.
    """
    rp_tuple = tuple([key.strip('_') for key in sorted(return_period_dict, key=return_period_dict.get, reverse=True) if key != 'None'] + [None])

    return rp_tuple 

def run_damage_reduction_by_asset(assets, geom_dict, overlay_assets, hazard_numpified_list, collect_inb_bl, changed_assets, hazard_intensity, fragility_values, maxdams_filt, map_rp_spec = None, asset_options=None, rp_spec_priority = None, reporting=True, adaptation_unit_cost=22500):
    """
    Calculates damages for assets under adapted conditions and computes adaptation costs at an asset-level.

    Args:
        assets (GeoDataFrame): Asset data.
        geom_dict (dict): Dictionary of asset geometries.
        overlay_assets (DataFrame): DataFrame of overlay assets.
        hazard_numpified_list (list): List of hazard intensity arrays. First array is the lower bound, second array is the upper bound.
        collect_inb_bl (dict): Baseline damage data.
        changed_assets (DataFrame): DataFrame of changed assets.
        hazard_intensity (ndarray): Hazard intensity values.
        fragility_values (ndarray): Vulnerability or fragility values.
        maxdams_filt (Series): Filtered replacement cost values.
        map_rp_spec (str, optional): Map return period specification. Defaults to None.
        asset_options (dict, optional): Dictionary of asset options including return period design. Defaults to None.
        rp_spec_priority (list, optional): List of return period priorities. Defaults to None.
        reporting (bool, optional): Whether to print reporting information. Defaults to True.
        adaptation_unit_cost (int, optional): Unit cost of adaptation. Defaults to 22500.

    Returns:
        tuple: Baseline damages, adapted damages, and adaptation costs.
    """
    # initialize dictionaries to hold the intermediate results
    collect_inb_adapt = {}
    adaptation_cost={}
    unchanged_assets = []

    if asset_options is not None and rp_spec_priority is not None and map_rp_spec is not None:
        skip_bridge, skip_tunnel = process_asset_options(asset_options, map_rp_spec, rp_spec_priority)
    else:
        skip_bridge=False
        skip_tunnel=False

    timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'{timestamp} - Calculating adapted damages for assets...')

    # interate over all unique assets and skip those that are not changed
    for asset in tqdm(overlay_assets.groupby('asset'),total=len(overlay_assets.asset.unique())): #asset is a tuple where asset[0] is the asset index or identifier and asset[1] is the asset-specific information
        if asset[0] not in changed_assets.index:
            unchanged_assets.append(asset[0])
            continue
        if changed_assets.loc[asset[0]].bridge == 'yes' and skip_bridge==True:
            collect_inb_adapt[asset[0]] = (0, 0)
            continue
        if changed_assets.loc[asset[0]].tunnel == 'yes' and skip_tunnel==True:
            collect_inb_adapt[asset[0]] = (0, 0)
            continue

        # retrieve asset geometry
        asset_geom = geom_dict[asset[0]]

        # calculate damages for the adapted conditions
        # - L1 adaptation
        # check hazard-level adaptation and spec, if asset adaptation spec is better than the map spec, asset is not damaged
        if changed_assets.loc[asset[0]].l1_adaptation is not None:
            asset_adapt_spec_index=rp_spec_priority.index(changed_assets.loc[asset[0]]['l1_rp_spec'])
            map_rp_spec_index=rp_spec_priority.index(map_rp_spec)
            if asset_adapt_spec_index <= map_rp_spec_index:
                collect_inb_adapt[asset[0]] = (0, 0)
                continue
        # - L2 adaptation
        # check asset-level adaptation, if None, asset is not modified
        if changed_assets.loc[asset[0]].l2_adaptation_exp is None and changed_assets.loc[asset[0]].l2_adaptation_vul is None:
            adaptation_cost[asset[0]]=0
            collect_inb_adapt[asset[0]]=collect_inb_bl[asset[0]]
            continue
            
        else:
            if changed_assets.loc[asset[0]].l2_adaptation_exp is None:
                h_mod=0
            else:
                h_mod=changed_assets.loc[asset[0]].l2_adaptation_exp #exposure modifier between 0 and the maximum hazard intensity
            hazard_numpified_list_mod = [np.array([[max(0.0, x[0] - h_mod), x[1]] for x in haz_numpified_bounds]) for haz_numpified_bounds in hazard_numpified_list]
            if changed_assets.loc[asset[0]].l2_adaptation_vul is None:
                v_mod=1
            else:
                v_mod=changed_assets.loc[asset[0]].l2_adaptation_vul #vulnerability modifier between invulnerable (0) and fully vulnerable(1)
            
            # calculate the adaptation cost
            get_hazard_points = hazard_numpified_list_mod[0][asset[1]['hazard_point'].values] 
            get_hazard_points[intersects(get_hazard_points[:,1],asset_geom)]
            
            if map_rp_spec == changed_assets.loc[asset[0]].l2_rp_spec: 
                if len(get_hazard_points) == 0: # no overlay of asset with hazard
                    affected_asset_length=0
                else:
                    if asset_geom.geom_type == 'LineString':
                        affected_asset_length = length(intersection(get_hazard_points[:,1],asset_geom)) # get the length of exposed meters per hazard cell
                adaptation_cost[asset[0]]=np.sum(affected_asset_length*adaptation_unit_cost) # calculate the adaptation cost in EUR Considering between 15 and 30 M based on Flyvbjerg et al (referring to Halcrow Fox 2000)
            else:
                adaptation_cost[asset[0]]=0
            
            number_of_lines = get_number_of_lines(assets.loc[asset[0]])
            if int(number_of_lines) == 2:
                double_track_factor = 0.5
            else: 
                double_track_factor = 1.0

            collect_inb_adapt[asset[0]] = tuple(ds.get_damage_per_asset(asset,h_numpified,asset_geom,hazard_intensity,fragility_values*v_mod,maxdams_filt, double_track_factor)[0] for h_numpified in hazard_numpified_list_mod)

    print(f'{len(unchanged_assets)} assets with no change.')

        #reporting
    if reporting==True:
        for asset_id, baseline_damages in collect_inb_bl.items():
            print(f'\nADAPTATION results for asset {asset_id}:')
            print(f'Baseline damages for asset {asset_id}: {baseline_damages[0]:.2f} to {baseline_damages[1]:.2f} EUR')
            print(f'Adapted damages for asset {asset_id}: {collect_inb_adapt[asset_id][0]:.2f} to {collect_inb_adapt[asset_id][1]:.2f} EUR')
            delta = tuple(collect_inb_adapt[asset_id][i] - baseline_damages[i] for i in range(len(baseline_damages)))
            # percent_change = tuple((100 * (delta[i] / baseline_damages[i])) for i in range(len(baseline_damages)))
            percent_change = tuple((100 * (delta[i] / baseline_damages[i])) if baseline_damages[i] != 0 else 0 for i in range(len(baseline_damages)))
            print(f'Change (Adapted-Baseline): {delta[0]:.2f} to {delta[1]:.2f} EUR, {percent_change}% change, at a cost of {adaptation_cost[asset_id]:.2f} EUR')

    return collect_inb_bl, collect_inb_adapt, adaptation_cost

def calculate_dynamic_return_periods(return_period_dict, num_years, increase_factor):
    """
    Calculates dynamic return periods over a specified number of years and calculates return periods in the future based on an increase factor.

    Args:
        return_period_dict (dict): Dictionary of return periods.
        num_years (int): Number of years.
        increase_factor (dict): Dictionary of increase factors.

    Returns:
        dict: Dynamic return periods.
    """
    years = np.linspace(0, num_years, num_years + 1)
    return_periods = {}
    for category, rp in return_period_dict.items(): 
        freq = 1 / rp
        freq_new = freq * increase_factor[category]
        freqs = np.interp(years, [0, num_years], [freq, freq_new])
        return_periods[category] = [1 / freq for freq in freqs]

    return return_periods

def ead_by_ts_plot(ead_by_ts):
    """
    Plots Expected Annual Damages (EAD) over time using Matplotlib.

    Args:
        ead_by_ts (DataFrame): DataFrame of EAD values over time.
    """
    import matplotlib.pyplot as plt
    plt.fill_between(ead_by_ts.index, ead_by_ts['Total Damage Lower Bound'], ead_by_ts['Total Damage Upper Bound'], alpha=0.3, color='red')
    plt.title('Expected Annual Damages (EAD) over time')
    plt.xlabel('Years from baseline')
    plt.ylabel('EAD (euros)')
    plt.legend(['Damage Bounds'], loc='upper left')
    plt.ylim(0)  # Set y-axis lower limit to 0
    plt.show()


def calculate_new_paths(graph_v, shortest_paths, disrupted_edges):
    """
    Calculates new shortest paths in a graph after removing disrupted edges.

    Args:
        graph_v (Graph): Graph representing the infrastructure network.
        shortest_paths (dict): Dictionary of shortest paths.
        disrupted_edges (list): List of disrupted edges.

    Returns:
        dict: Dictionary of new shortest paths.
    """
    graph_v_disrupted=graph_v.copy()
    for u,v in set(disrupted_edges):
        graph_v_disrupted.remove_edge(u,v,0)
        
    disrupted_shortest_paths={}
    for (origin,destination), (nodes_in_spath,demand) in shortest_paths.items():
        edges_in_spath=[(nodes_in_spath[i],nodes_in_spath[i+1]) for i in range(len(nodes_in_spath)-1)]
        if set(disrupted_edges).isdisjoint(edges_in_spath):
            continue
        else:
            try:
                disrupted_shortest_paths[(origin,destination)] = (nx.shortest_path(graph_v_disrupted, origin, destination, weight='weight'), demand)
            except nx.NetworkXNoPath:
                print(f'No path between {origin} and {destination}. Cannot ship by train.')
                disrupted_shortest_paths[(origin,destination)] = (None, demand)
                continue
    
    return disrupted_shortest_paths

def calculate_economic_impact_shortest_paths(hazard_map, graph, shortest_paths, disrupted_shortest_paths, average_train_load_tons, average_train_cost_per_ton_km, average_road_cost_per_ton_km):
    """
    Computes the economic impact of disrupted shortest paths based on cost of shipping by road and additional distance travelled by train.

    Args:
        hazard_map (str): Identifier of the hazard map.
        graph (Graph): Graph representing the infrastructure network.
        shortest_paths (dict): Dictionary of shortest paths.
        disrupted_shortest_paths (dict): Dictionary of shortest paths under disrupted conditions.
        average_train_load_tons (float): Average train load in tons.
        average_train_cost_per_ton_km (float): Average train cost per ton-kilometer.
        average_road_cost_per_ton_km (float): Average road cost per ton-kilometer.

    Returns:
        float: Economic impact of the disruptions.
    """
    # hazard_map = 'flood_DERP_RW_L_4326_2080430320'
    haz_rp=hazard_map.split('_RW_')[-1].split('_')[0]

    #duration of disruption = 1 week for haz_rp 'H', 2 for 'M' and 10 for 'L'
    duration_dict={'H':1, 'M':2, 'L':10}
    duration=duration_dict[haz_rp]
    economic_impact = 0
    # Loop through the edges where there is a change in flow
    for (origin, destination), (nodes_in_path, demand) in disrupted_shortest_paths.items():
        # Find the length of the initial shortest path
        length_old_path=0
        for i in range(len(shortest_paths[(origin, destination)][0])-1):
            length_old_path += graph.edges[shortest_paths[(origin, destination)][0][i], shortest_paths[(origin, destination)][0][i+1], 0]['length']/1000

        # If there is no path available, calculate cost of shipping by road             
        if (nodes_in_path is None) or ('_d' in str(nodes_in_path)):
            economic_impact += duration*demand*average_train_load_tons*(average_road_cost_per_ton_km-average_train_cost_per_ton_km)*length_old_path
            continue

        # If there is a path available, find the length of the new shortest path and find the cost due to additional distnce travelled
        else:
            length_new_path=0
            for i in range(len(nodes_in_path)-1):
                length_new_path += graph.edges[nodes_in_path[i], nodes_in_path[i+1], 0]['length']/1000
            economic_impact += duration*demand*average_train_load_tons*average_train_cost_per_ton_km*(length_new_path-length_old_path)
        
    # returns the economic impact for an infrastructure region given the infrastructure graph and shortest paths between ods, a set of disrupted shortest paths and the average train loads and costs
    return economic_impact

def _inspect_graph(graph):
    """
    Inspects the types of edge capacities, edge weights, and node demands in a graph to ensure they are integers - floats slow flow computations.

    Args:
        graph (Graph): Graph representing the infrastructure network.

    Returns:
        tuple: Lists of types for edge capacities, edge weights, and node demands.
    """
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

def create_virtual_graph(graph):
    """
    Creates a virtual graph with dummy nodes and edges to simulate maximum capacities and weights.
    Adapted from code by Asgarpour, S.

    Args:
        graph (Graph): Graph representing the infrastructure network.

    Returns:
        Graph: Virtual graph with dummy nodes and edges.
    """
    max_weight_graph = max(attr['weight'] for _, _, attr in graph.edges(data=True))
    print('Max weight: '+str(max_weight_graph))
    max_capacity_graph = 1#max(attr['capacity'] for _, _, attr in graph.edges(data=True))
    print('Max capacity: '+str(max_capacity_graph))

    # create a virtual node with dummy nodes
    graph_v=graph.copy()
    # convert to int
    for u, v, key, attr in graph.edges(keys=True, data=True):
        graph_v.add_edge((str(u) + '_d'), (str(v) + '_d'), **attr)

    for u in graph.nodes:
        graph_v.add_edge(u,(str(u) + '_d'),capacity=max_capacity_graph*100,weight=int(round(1e10,0)))
        graph_v.add_edge((str(u) + '_d'),u,capacity=max_capacity_graph*100,weight=0)

    # verify capacities, weights and demands are integers
    edge_capacities_types, edge_weights_types, node_demands_types = _inspect_graph(graph_v)

    if {type(int())} == set(list(edge_capacities_types) + list(edge_weights_types) + list(node_demands_types)):
        print('Success: only int type values')
    else: 
        print('Warning! Not all values are integers')

    return graph_v

# Assign a weight to each edge based on the length of the edge
def set_edge_weights(assets, graph):
    """
    Assigns weights to graph edges based on the lengths of the corresponding assets.

    Args:
        assets (GeoDataFrame): Asset data.
        graph (Graph): Graph representing the infrastructure network.

    Returns:
        Graph: Graph with updated edge weights. Since weights must be integers, the length is multiplied by 1e3 and rounded.
    """
    # Create a dictionary to store the length of each asset
    asset_lengths = {str(asset['osm_id']): asset['geometry'].length for asset_id, asset in assets.iterrows()}

    # Loop through the edges and assign the length of the asset to the edge
    for u, v, attr in graph.edges(data=True):
        if 'source_sink' in str(u) or 'source_sink' in str(v):
            continue

        # Initialize the weight and length of the edge
        attr['weight'] = int(0)
        attr['length'] = 0
        if 'osm_id' not in attr:
            continue
        
        # For concatenated edges, split the osm_id string and sum the lengths for each asset
        osm_ids = attr['osm_id'].split('; ')
        for osm_id in osm_ids:
            if osm_id in asset_lengths:
                attr['length'] += asset_lengths[osm_id]
                attr['weight'] += int(round(asset_lengths[osm_id]*1e3,0))

    return graph

def _create_terminal_graph(graph):
    """
    Creates a subgraph containing only terminal nodes.

    Args:
        graph (Graph): Graph representing the infrastructure network.

    Returns:
        Graph: Subgraph with only terminal nodes.
    """
    terminal_graph = graph.copy()
    for u, attr in graph.nodes(data=True):
        # If the node has no possible_terminal attributes indicated, skip it and use as possible terminal
        if 'possible_terminal' not in graph.nodes[u]:
            continue
        if attr['possible_terminal'] == 0: 
            terminal_graph.remove_node(u)
    print('Possible terminals:', terminal_graph.number_of_nodes())
    
    return terminal_graph

def shortest_paths_between_terminals(graph, route_data):
    """
    Finds the shortest paths between terminal nodes in a graph based on route data.

    Args:
        graph (Graph): Graph representing the infrastructure network.
        route_data (DataFrame): DataFrame of route data.

    Returns:
        dict: Dictionary of shortest paths between terminal nodes.
    """
    # Make a copy of the graph with only the nodes identified as possible terminals
    terminal_graph = _create_terminal_graph(graph)

    # Create a dictionary to store the shortest path between each OD pair
    paths={}

    # Iterate over all route ODs pairs and find the shortest path between the two nodes
    # for _, attr in route_data.iterrows():
    fail_count=0
    for _, attr in tqdm(route_data.iterrows(), total=route_data.shape[0], desc='Finding shortest paths between origin-destination pairs'):
        # Snap route origin and destination geometries to nearest terminal node on graph
        if attr['geometry_from'].geom_type == 'Point':
            centroid_from = attr['geometry_from']
        else:
            centroid_from = attr['geometry_from'].centroid
        from_nearest_node = nearest_nodes(terminal_graph, centroid_from, 1)
        if attr['geometry_to'].geom_type == 'Point':
            centroid_to = attr['geometry_to']
        else:
            centroid_to = attr['geometry_to'].centroid
        to_nearest_node = nearest_nodes(terminal_graph, centroid_to, 1)

        # If the nearest nodes are the same for the origin and destination, skip the route
        if from_nearest_node[0][0] == to_nearest_node[0][0]:
            continue
        # Add name to node in graph
        if 'name' not in graph.nodes[from_nearest_node[0][0]]:
            graph.nodes[from_nearest_node[0][0]]['name']=attr['From']
        else:
            if graph.nodes[from_nearest_node[0][0]]['name']!=attr['From']:
                print(f'Name mismatch: {graph.nodes[from_nearest_node[0][0]]["name"]} vs {attr["From"]}, updating to {attr["From"]}')
                graph.nodes[from_nearest_node[0][0]]['name']=attr['From']
            else: 
                pass

        if 'name' not in graph.nodes[to_nearest_node[0][0]]:
            graph.nodes[to_nearest_node[0][0]]['name']=attr['To']
        else:   
            if graph.nodes[to_nearest_node[0][0]]['name']!=attr['To']:
                print(f'Name mismatch: {graph.nodes[to_nearest_node[0][0]]["name"]} vs {attr["To"]}, updating to {attr["To"]}')
                graph.nodes[to_nearest_node[0][0]]['name']=attr['To']
            else: 
                pass                

        # Find the shortest path between the two terminals and the average flow on the path
        try:
            shortest_path = nx.shortest_path(graph, from_nearest_node[0][0], to_nearest_node[0][0], weight='weight')
            paths[(from_nearest_node[0][0], to_nearest_node[0][0])] = (shortest_path, int(ceil(attr['goods']/52)))
            #here add name to node from route data    
        except nx.NetworkXNoPath:
            fail_count += 1
            continue               
    print(f'Failed to find paths for {fail_count} routes')
    return paths

def prepare_route_data(route_data_source, assets=None):
    """
    Prepares route data by filtering and converting it to geometries, optionally filtering by asset bounds.

    Args:
        route_data_source (str): Path to the route data source file.
        assets (GeoDataFrame, optional): Asset data. Defaults to None.

    Returns:
        DataFrame: Prepared route data.
    """
    route_data = pd.read_excel(route_data_source)
    transformer=Transformer.from_crs("EPSG:4326", "EPSG:3857")

    # Load route data
    route_data = pd.read_excel(route_data_source)
    # Only keep columns that are necessary: From_Latitude, From_Longitude, To_Latitude, To_Longitude, Number_Goods_trains, Country
    route_data = route_data[['From', 'To', 'From_Latitude', 'From_Longitude', 'To_Latitude', 'To_Longitude', 'Number_Goods_trains', 'Country']]
    # Rename columns Number_Goods_trains to goods 
    route_data = route_data.rename(columns={'Number_Goods_trains' : 'goods'})
    # Drop rows with no goods
    route_data = route_data[route_data['goods'] > 0]
    # Drop rows that are not from Country "DE"
    route_data = route_data[route_data['Country'] == 'DE']
    # Convert From_Latitude, From_Longitude and To_Latitude, To_Longitude to geometries
    route_data['geometry_from'] = route_data.apply(lambda k: Point(k['From_Longitude'], k['From_Latitude']), axis=1)
    route_data['geometry_to'] = route_data.apply(lambda k: Point(k['To_Longitude'], k['To_Latitude']), axis=1)

    if assets is None:
            # # Reproject geometries of points from 4326 to 3857
        route_data['geometry_from'] = route_data['geometry_from'].apply(lambda k: Point(transformer.transform(k.y, k.x)))
        route_data['geometry_to'] = route_data['geometry_to'].apply(lambda k: Point(transformer.transform(k.y, k.x)))

        return route_data
    
    # Filter route data to only include routes that are within the bounds of the assets
    assets_bounds=assets.copy().to_crs(4326).total_bounds
    route_data = route_data[route_data['geometry_from'].apply(lambda geom: box(*assets_bounds).contains(geom))]
    route_data = route_data[route_data['geometry_to'].apply(lambda geom: box(*assets_bounds).contains(geom))]
    # # Reproject geometries of points from 4326 to 3857
    route_data['geometry_from'] = route_data['geometry_from'].apply(lambda k: Point(transformer.transform(k.y, k.x)))
    route_data['geometry_to'] = route_data['geometry_to'].apply(lambda k: Point(transformer.transform(k.y, k.x)))

    return route_data

def set_edge_capacities(graph, route_data, simplified=False):
    """
    Sets capacities for graph edges based on shortest paths between origin-destination pairs.

    Args:
        graph (Graph): Graph representing the infrastructure network.
        route_data (DataFrame): DataFrame of route data.
        simplified (bool, optional): Whether to use simplified capacity assignment, (boolean, 1=available). Defaults to False.

    Returns:
        tuple: Graph with updated capacities and dictionary of shortest paths.
    """    
    paths=shortest_paths_between_terminals(graph, route_data)
    
    if simplified==True:
        for _,_, attr in graph.edges(data=True):
            if 'capacity' not in attr:
                attr['capacity'] = 1
        
        return graph, paths

    # Assign capacity to edges that are part of a shortest path
    for (_,_), (nodes_in_path,average_flow) in paths.items():
        for i in range(len(nodes_in_path)-1):
            if not graph.has_edge(nodes_in_path[i], nodes_in_path[i+1], 0):
                continue
            if nodes_in_path[i]=='source_sink' or nodes_in_path[i+1]=='source_sink':
                continue 
            if 'capacity' in graph[nodes_in_path[i]][nodes_in_path[i+1]][0]:
                graph[nodes_in_path[i]][nodes_in_path[i+1]][0]['capacity'] = max(graph[nodes_in_path[i]][nodes_in_path[i+1]][0]['capacity'],2*average_flow)
            else:
                graph[nodes_in_path[i]][nodes_in_path[i+1]][0]['capacity'] = 2*average_flow
    
    # Set the capacity of edges that are not on a shortest path to the median capacity
    caps=[attr['capacity'] for _, _, attr in graph.edges(data=True) if 'capacity' in attr]
    median_cap = int(np.median(caps))
    for _,_, attr in graph.edges(data=True):
        if 'capacity' not in attr:
            attr['capacity'] = median_cap
        
    return graph, paths

def nearest_nodes(graph, point, n):
    """
    Finds the nearest nodes in a graph to a given point.

    Args:
        graph (Graph): Graph representing the infrastructure network.
        point (Point): Point to find the nearest nodes to.
        n (int): Number of nearest nodes to find.

    Returns:
        list: List of nearest nodes and their distances.
    """
    near_nodes = []
    for node, attr in graph.nodes(data=True):
        if 'geometry' in attr:
            distance = point.distance(attr['geometry'])
            near_nodes.append((node, distance))
    
    near_nodes = sorted(near_nodes, key=lambda x: x[1])

    return near_nodes[:n]

def recalculate_disrupted_edges(graph_v, assets, disrupted_edges, fully_protected_assets, unexposed_osm_ids):
    """
    Recalculates disrupted edges in a graph considering fully protected and unexposed assets.

    Args:
        G_v (Graph): Graph representing the infrastructure network.
        assets (GeoDataFrame): Asset data.
        disrupted_edges (list): List of disrupted edges.
        fully_protected_assets (list): List of fully protected asset indices.
        unexposed_osm_ids (list): List of unexposed OSM IDs.

    Returns:
        list: List of adapted disrupted edges.
    """
    # list of osm_ids of adapted assets
    adapted_osm_ids=assets.loc[assets.index.isin(fully_protected_assets)]['osm_id'].values
    available_osm_ids = np.unique(np.concatenate((unexposed_osm_ids, adapted_osm_ids)))
    available_edges=[]
    # loop through the disrupted edges to check if previously disrupted edges are now available
    for (u,v) in disrupted_edges:
        # get the attributes of the edge
        osm_ids_edge = graph_v.edges[(u,v,0)]['osm_id'].split(';')
        osm_ids_edge = [ids.strip() for ids in osm_ids_edge]

        # check if all the osm_ids of the edge are in the list of adapted assets
        if set(osm_ids_edge).issubset(available_osm_ids):
            available_edges.append((u,v))
        
    adapted_disrupted_edges = [edge for edge in disrupted_edges if edge not in available_edges]

    return adapted_disrupted_edges

def filter_assets_to_adapt(assets, adaptation_area):
    """
    Filters assets that need adaptation based on specified adaptation areas.

    Args:
        assets (GeoDataFrame): Asset data.
        adaptation_area (GeoDataFrame): Adaptation area data, including protected geometry, adaptation level, and return period specification.

    Returns:
        GeoDataFrame: Filtered assets to adapt.
    """
    assets_to_adapt = gpd.GeoDataFrame()
    if len(adaptation_area)==0:
        return assets_to_adapt
    
    filtered_adaptation_area = adaptation_area[adaptation_area['geometry'].notnull()]
    for (adaptation_id, ad) in filtered_adaptation_area.iterrows():
        adaptation = gpd.GeoDataFrame(ad).T
        adaptation = adaptation.set_geometry('geometry').set_crs(3857)
        filtered_assets = gpd.overlay(assets, adaptation, how='intersection')
        a_assets = assets.loc[(assets['osm_id'].isin(filtered_assets['osm_id']))].copy().drop(columns=['other_tags'])
        a_assets.loc[:, 'adaptation_id'] = adaptation_id
        a_assets.loc[:, 'prot_area'] = adaptation['prot_area'].values[0]
        a_assets.loc[:, 'adapt_level'] = adaptation['adapt_level'].values[0]        
        a_assets.loc[:, 'rp_spec'] = adaptation['rp_spec'].values[0].upper()
        a_assets.loc[:, 'adapt_size'] = adaptation['adapt_size'].values[0]
        a_assets.loc[:, 'adapt_unit'] = adaptation['adapt_unit'].values[0]
        assets_to_adapt = pd.concat([assets_to_adapt, a_assets], ignore_index=False)

    return assets_to_adapt

def load_baseline_run(hazard_map, interim_data_path, only_overlay=False):
    """
    Loads baseline run data for a hazard map, consisting of hazard-asset overlays and hazard intensity data.

    Args:
        hazard_map (str): Hazard map identifier.
        interim_data_path (Path): Path to interim data storage.
        only_overlay (bool, optional): Whether to load only the overlay data. Defaults to False.

    Returns:
        tuple: Overlay assets and hazard intensity data.
    """
    parts = hazard_map.split('_')
    try:
        bas = parts[-1]  # Assuming the return period is the last part
        rp = parts[-3]  # Assuming the basin is the third to last part
    except:
        print("Invalid hazard_map format")
    
    # open pickled hazard-asset overlay and hazard intensity data
    with open(interim_data_path / f'overlay_assets_flood_DERP_RW_{rp}_4326_{bas}.pkl', 'rb') as f:
        overlay_assets = pickle.load(f)

    if only_overlay:
        return overlay_assets    
    with open(interim_data_path / f'hazard_numpified_flood_DERP_RW_{rp}_4326_{bas}.pkl', 'rb') as f:
        hazard_numpified_list = pickle.load(f)
    
    return overlay_assets, hazard_numpified_list

def run_direct_damage_reduction_by_hazmap(assets, geom_dict, overlay_assets, hazard_numpified_list, collect_inb_bl, adapted_assets, map_rp_spec=None, asset_options=None, rp_spec_priority = None, reporting=False, adaptation_unit_cost=22500):
    """
    Runs direct damage reduction analysis for a hazard map, calculating damages and adaptation costs.

    Args:
        assets (GeoDataFrame): Asset data.
        geom_dict (dict): Dictionary of asset geometries.
        overlay_assets (DataFrame): DataFrame of overlay assets.
        hazard_numpified_list (list): List of hazard intensity arrays.
        collect_inb_bl (dict): Baseline damage data.
        adapted_assets (DataFrame): DataFrame of adapted assets.
        map_rp_spec (str, optional): Map return period specification. Defaults to None.
        asset_options (dict, optional): Dictionary of asset options. Defaults to None.
        rp_spec_priority (list, optional): List of return period priorities. Defaults to None.
        reporting (bool, optional): Whether to print reporting information. Defaults to False.
        adaptation_unit_cost (int, optional): Unit cost of adaptation. Defaults to 22500.

    Returns:
        tuple: Adaptation run results.
    """
    data_path = Path(pathlib.Path.home().parts[0]) / 'Data'
    # Load configuration with ini file (created running config.py)
    config_file=r'C:\repos\ci_adapt\config_ci_adapt.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    hazard_type = config.get('DEFAULT', 'hazard_type')
    infra_type = config.get('DEFAULT', 'infra_type')
    vulnerability_data = config.get('DEFAULT', 'vulnerability_data')
    infra_curves, maxdams = ds.read_vul_maxdam(data_path, hazard_type, infra_type)
    max_damage_tables = pd.read_excel(data_path / vulnerability_data / 'Table_D3_Costs_V1.0.0.xlsx',sheet_name='Cost_Database',index_col=[0])
    print(f'Found matching infrastructure curves for: {infra_type}')

    hazard_intensity = infra_curves['F8.1'].index.values
    fragility_values = (np.nan_to_num(infra_curves['F8.1'].values,nan=(np.nanmax(infra_curves['F8.1'].values)))).flatten()
    maxdams_filt=max_damage_tables[max_damage_tables['ID number']=='F8.1']['Amount']
    print('-- Calculating direct damages --')
    adaptation_run = run_damage_reduction_by_asset(assets, geom_dict, overlay_assets, hazard_numpified_list, collect_inb_bl, adapted_assets, hazard_intensity, fragility_values, maxdams_filt, 
                                                   map_rp_spec=map_rp_spec, asset_options=asset_options, rp_spec_priority = rp_spec_priority, reporting=reporting, adaptation_unit_cost=adaptation_unit_cost)

    return adaptation_run       

def run_indirect_damages_by_hazmap(adaptation_run, assets, hazard_map, overlay_assets, disrupted_edges, shortest_paths, graph_v, average_train_load_tons, average_train_cost_per_ton_km, average_road_cost_per_ton_km):
    """
    Runs indirect damage analysis for a hazard map, calculating economic impacts of disrupted paths.

    Args:
        adaptation_run (tuple): Results of the adaptation run.
        assets (GeoDataFrame): Asset data.
        hazard_map (str): Hazard map identifier.
        overlay_assets (DataFrame): DataFrame of overlay assets.
        disrupted_edges (list): List of disrupted edges.
        shortest_paths (dict): Dictionary of shortest paths.
        graph_v (Graph): Graph representing the infrastructure network.
        average_train_load_tons (float): Average train load in tons.
        average_train_cost_per_ton_km (float): Average train cost per ton-kilometer.
        average_road_cost_per_ton_km (float): Average road cost per ton-kilometer.

    Returns:
        float: Economic impact of the disruptions.
    """
    # For a given hazard map overlay, find all the assets that are fully protected
    fully_protected_assets=[asset_id for asset_id, damages in adaptation_run[1].items() if damages[0]==0 and damages[1]==0]

    # For a given hazard map overlay, find all assets that are not exposed to flooding
    unexposed_assets=[asset_id for asset_id in assets.index if asset_id not in overlay_assets.asset.values]
    unexposed_osm_ids=assets.loc[assets.index.isin(unexposed_assets)]['osm_id'].values

    # find the disrupted edges and paths under adapted conditions
    print('-- Calculating indirect damages --')
    # find edges that will no longer be disrupted
    print('disrupted_edges baseline: ', disrupted_edges)
    disrupted_edges_adapted = recalculate_disrupted_edges(graph_v, assets, disrupted_edges, fully_protected_assets, unexposed_osm_ids)
    print('disrupted_edges_adapted: ', disrupted_edges_adapted)

    disrupted_shortest_paths_adapted=calculate_new_paths(graph_v, shortest_paths, disrupted_edges_adapted)

    if disrupted_shortest_paths_adapted == {}: # No disrupted paths, no economic impact
        print(f'No shortest paths disrupted for {hazard_map}. No economic impact.')
        return 0

    impact=calculate_economic_impact_shortest_paths(hazard_map, graph_v, shortest_paths, disrupted_shortest_paths_adapted, average_train_load_tons, average_train_cost_per_ton_km, average_road_cost_per_ton_km)
    print(hazard_map, impact)
    return impact


def add_l1_adaptation(adapted_assets, affected_assets, rp_spec_priority):
    """
    Adds level 1 adaptation to assets based on protection areas and return period specifications.

    Args:
        adapted_assets (DataFrame): DataFrame of adapted assets.
        affected_assets (DataFrame): DataFrame of affected assets.
        rp_spec_priority (list): List of return period priorities.

    Returns:
        DataFrame: Updated adapted assets.
    """
    for asset_id in affected_assets.index:
        current_prio=rp_spec_priority.index(adapted_assets.loc[asset_id]['l1_adaptation'])
        adaptation_prio=rp_spec_priority.index(affected_assets.loc[asset_id]['rp_spec'])
        if adaptation_prio < current_prio:
            adapted_assets.loc[asset_id, 'l1_adaptation'] = affected_assets.loc[asset_id]['prot_area']
            adapted_assets.loc[asset_id, 'l1_rp_spec'] = affected_assets.loc[asset_id]['rp_spec']
             
    return adapted_assets

def add_l2_adaptation(adapted_assets, affected_assets, overlay_assets, hazard_numpified_list):
    """
    Adds level 2 adaptation to assets, modifying exposure and vulnerability.

    Args:
        adapted_assets (DataFrame): DataFrame of adapted assets.
        affected_assets (DataFrame): DataFrame of affected assets.
        overlay_assets (DataFrame): DataFrame of overlay assets.
        hazard_numpified_list (list): List of hazard intensity arrays.

    Returns:
        DataFrame: Updated adapted assets.
    """
    final_red = {}
    red = affected_assets['adapt_unit'].values[0]
    for asset_id in affected_assets.index:
        if red=='exp_red':
            if adapted_assets.loc[asset_id]['l2_adaptation_exp'] == None:
                current_red = 0
            else:
                current_red = adapted_assets.loc[asset_id]['l2_adaptation_exp']
            max_int_haz_map=retrieve_max_intensity_by_asset(asset_id, overlay_assets, hazard_numpified_list)
            if len(max_int_haz_map)==0:
                max_int_haz_map=[0]
            if np.max(max_int_haz_map) > current_red:
                final_red[asset_id] = np.max(max_int_haz_map)
        elif red=='vul_red':
            if adapted_assets.loc[asset_id]['l2_adaptation_vul'] == None:
                current_red = 0
            else:
                current_red = adapted_assets.loc[asset_id]['l2_adaptation_vul']
            if current_red > affected_assets.loc[asset_id]['adapt_size']:
                final_red[asset_id] = affected_assets.loc[asset_id]['adapt_size']            
            print('Vulnerability reduction not tested yet')
        elif red=='con_red':
            print('Consequence reduction not implemented yet')
        else: 
            print('Adaptation not recognized, for l2 adaptation exposure, vulnerability, or consequence reduction must be specified (exp_red, vul_red, con_red)')
    
    if red=='exp_red':
        for asset_id in final_red.keys():
            adapted_assets.loc[asset_id, 'l2_adaptation_exp'] = final_red[asset_id]
            adapted_assets.loc[asset_id, 'l2_rp_spec'] = affected_assets.loc[asset_id]['rp_spec']
    elif red=='vul_red':
        for asset_id in final_red.keys():
            adapted_assets.loc[asset_id, 'l2_adaptation_vul'] = final_red[asset_id]
            adapted_assets.loc[asset_id, 'l2_rp_spec'] = affected_assets.loc[asset_id]['rp_spec']
    
    return adapted_assets

def find_edges_by_osm_id_pair(graph_v, osm_id_pair):
    """
    Finds edges in a graph that contain specified OSM IDs.

    Args:
        graph_v (Graph): Graph representing the infrastructure network.
        osm_id_pair (tuple): Pair of OSM IDs.

    Returns:
        tuple: Edges containing the specified OSM IDs.
    """
    osm_id1, osm_id2 = osm_id_pair
    edge1 = [(u,v) for u,v,key,attr in graph_v.edges(keys=True, data=True) if 'osm_id' in attr and str(osm_id1) in attr['osm_id']][0]
    edge2 = [(u,v) for u,v,key,attr in graph_v.edges(keys=True, data=True) if 'osm_id' in attr and str(osm_id2) in attr['osm_id']][0]
    return edge1, edge2

def find_closest_nodes(graph_v, edge1, edge2):
    """
    Finds the closest nodes between two edges in a graph.

    Args:
        G_v (Graph): Graph representing the infrastructure network.
        edge1 (tuple): First edge.
        edge2 (tuple): Second edge.

    Returns:
        list: Closest nodes between the two edges.
    """
    u1 = edge1[0]
    v1 = edge1[1]
    u2 = edge2[0]
    v2 = edge2[1]

    dist_u1_u2 = graph_v.nodes[u1]['geometry'].distance(graph_v.nodes[u2]['geometry'])
    dist_u1_v2 = graph_v.nodes[u1]['geometry'].distance(graph_v.nodes[v2]['geometry'])
    dist_v1_u2 = graph_v.nodes[v1]['geometry'].distance(graph_v.nodes[u2]['geometry'])
    dist_v1_v2 = graph_v.nodes[v1]['geometry'].distance(graph_v.nodes[v2]['geometry'])

    dists = [dist_u1_u2, dist_u1_v2, dist_v1_u2, dist_v1_v2]
    min_dist = min(dists)

    closest_nodes_index = dists.index(min_dist)
    if closest_nodes_index == 0:
        closest_nodes = [u1, u2]
    elif closest_nodes_index == 1:
        closest_nodes = [u1, v2]
    elif closest_nodes_index == 2:
        closest_nodes = [v1, u2]
    elif closest_nodes_index == 3:
        closest_nodes = [v1, v2]
    
    return closest_nodes

def add_l3_adaptation(graph_v, osm_id_pair, detour_index=0.5, adaptation_unit_cost=3700*10): #detour index is the ratio of direct distance / transport distance, rugged topography has a lower detour index while flat topography are closer to 1.
    """
    Adds a level 3 adaptation by creating new connections between assets in a graph.

    Args:
        graph_v (Graph): Graph representing the infrastructure network.
        osm_id_pair (tuple): Pair of OSM IDs representing the assets to connect.
        detour_index (float, optional): Ratio of direct distance to transport distance. Defaults to 0.5.
        adaptation_unit_cost (float, optional): Cost per unit length of the adaptation. Defaults to 3700*10.

    Returns:
        tuple: Updated graph and the adaptation cost.
    """
    edge1, edge2 = find_edges_by_osm_id_pair(graph_v, osm_id_pair)
    
    closest_nodes = find_closest_nodes(graph_v, edge1, edge2)
    geom_01 = [shapely.LineString([graph_v.nodes[closest_nodes[0]]['geometry'], graph_v.nodes[closest_nodes[1]]['geometry']])]
    geom_10 = [shapely.LineString([graph_v.nodes[closest_nodes[1]]['geometry'], graph_v.nodes[closest_nodes[0]]['geometry']])]
    length_01 = geom_01[0].length
    length_10 = geom_10[0].length
    graph_v.add_edge(closest_nodes[0], closest_nodes[1], osm_id='l3_adaptation_to', capacity=1, weight=int(round(length_01*1e3/detour_index,0)), length=length_01, geometry=geom_01)
    graph_v.add_edge(closest_nodes[1], closest_nodes[0], osm_id='l3_adaptation_from', capacity=1, weight=int(round(length_10*1e3/detour_index,0)), length=length_10, geometry=geom_10)
    
    adaptation_cost = length_01*adaptation_unit_cost
    
    print('Applying adaptation: new connection between assets with osm_id ', osm_id_pair)
    print('Level 3 adaptation')
    return graph_v, adaptation_cost

def add_adaptation_columns(adapted_assets):
    """
    Adds columns for different levels of adaptation to the adapted assets DataFrame.

    Args:
        adapted_assets (DataFrame): DataFrame containing the adapted assets.

    Returns:
        DataFrame: Updated DataFrame with new adaptation columns set to None.
    """
    columns_to_set_none = [
        'l1_adaptation', 'l1_rp_spec', 'l2_adaptation_exp', 'l2_adaptation_vul', 
        'l2_rp_spec', 'l3_adaptation', 'l3_rp_spec', 'l4_adaptation', 'l4_rp_spec'
    ]

    for column in columns_to_set_none:
        adapted_assets[column] = None

    return adapted_assets

def create_adaptation_df(adapted_area):
    """
    Creates a DataFrame to store adaptation data for specified areas.

    Args:
        adapted_area (GeoDataFrame): GeoDataFrame containing the adapted areas.

    Returns:
        DataFrame: DataFrame with adaptation data.
    """
    adaptation_df_columns = ['id', 'prot_area', 'adapt_level', 'rp_spec', 'adaptation_cost']
    adaptation_df = pd.DataFrame(columns=adaptation_df_columns)

    if adapted_area is None:
        return adaptation_df
    
    adaptation_df['id'] = adapted_area.index.values
    adaptation_df.set_index('id', inplace=True)
    adaptation_df['prot_area'] = adapted_area['prot_area'].values
    adaptation_df['adapt_level'] = adapted_area['adapt_level'].values
    adaptation_df['rp_spec'] = adapted_area['rp_spec']
    
    return adaptation_df

def apply_asset_adaptations_in_haz_area(adapted_assets, affected_assets, overlay_assets, hazard_numpified_list, rp_spec_priority=None):
    """
    Applies specified adaptations to assets in a hazard area.

    Args:
        adapted_assets (DataFrame): DataFrame containing the adapted assets.
        affected_assets (DataFrame): DataFrame containing the affected assets.
        overlay_assets (DataFrame): DataFrame containing the asset-hazard overlay data.
        hazard_numpified_list (list): List of hazard intensity data, with the first element being the lower bound and the last element being the upper bound.
        rp_spec_priority (list, optional): List of return period specifications in priority order. Defaults to None.

    Returns:
        DataFrame: Updated DataFrame with applied adaptations.
    """
    print('Applying adaptation: ', affected_assets['prot_area'].values[0])
    if set(affected_assets['adapt_level'].values) == {1}:
        print('Level 1 adaptation')
        adapted_assets = add_l1_adaptation(adapted_assets, affected_assets, rp_spec_priority)
    elif set(affected_assets['adapt_level'].values) == {2}:
        print('Level 2 adaptation')
        adapted_assets = add_l2_adaptation(adapted_assets, affected_assets, overlay_assets, hazard_numpified_list)
    else:
        print('Adaptation level not recognized')
    
    return adapted_assets

def overlay_hazard_adaptation_areas(df_ds,adaptation_areas): #adapted from Koks
    """
    Overlays hazard data with adaptation areas to identify intersecting regions.

    Args:
        df_ds (GeoDataFrame): GeoDataFrame containing the hazard data.
        adaptation_areas (GeoDataFrame): GeoDataFrame containing the adaptation areas.

    Returns:
        DataFrame: DataFrame with intersecting regions.
    """
    hazard_tree = shapely.STRtree(df_ds.geometry.values)
    if (shapely.get_type_id(adaptation_areas.iloc[0].geometry) == 3) | (shapely.get_type_id(adaptation_areas.iloc[0].geometry) == 6): # id types 3 and 6 stand for polygon and multipolygon
        return  hazard_tree.query(adaptation_areas.geometry,predicate='intersects')    
    else:
        return  hazard_tree.query(adaptation_areas.buffered,predicate='intersects')

def get_cost_per_area(adapt_area,hazard_numpified,adapt_area_geom, adaptation_unit_cost=1): #adapted from Koks
    """
    Calculates the cost of adaptation for a specified area based on hazard overlays.

    Args:
        adapt_area (DataFrame): DataFrame containing the adaptation area.
        hazard_numpified (ndarray): Numpy array containing hazard intensity data.
        adapt_area_geom (Geometry): Geometry of the adaptation area.
        adaptation_unit_cost (float, optional): Cost per unit length of the adaptation. Defaults to 1.

    Returns:
        float: Total adaptation cost for the area.
    """
    # find the exact hazard overlays:
    get_hazard_points = hazard_numpified[adapt_area[1]['hazard_point']]#.values]
    get_hazard_points[shapely.intersects(get_hazard_points[:,1],adapt_area_geom)]
    # estimate damage
    if len(get_hazard_points) == 0: # no overlay of asset with hazard
        return 0
    else:
        if adapt_area_geom.geom_type == 'LineString':
            overlay_meters = shapely.length(shapely.intersection(get_hazard_points[:,1],adapt_area_geom)) # get the length of exposed meters per hazard cell
            return np.sum((np.float16(get_hazard_points[:,0]))*overlay_meters*adaptation_unit_cost) #return asset number, total damage for asset number (damage factor * meters * max. damage)
        else:
            print('Adaptation area not recognized')
            return 0

def process_adap_dat(single_footprint, adaptation_areas, hazard_numpified_list, adaptation_unit_cost=1.0):
    """
    Processes adaptation data for a hazard footprint, calculating adaptation costs for specified areas.

    Args:
        single_footprint (Path): Path to the hazard footprint file.
        adaptation_areas (GeoDataFrame): GeoDataFrame containing the adaptation areas.
        hazard_numpified_list (list): List of hazard intensity data.
        adaptation_unit_cost (float, optional): Cost per unit length of the adaptation. Defaults to 1.0.

    Returns:
        dict: Dictionary with adaptation costs for each area.
    """
    # load hazard map
    hazard_map = ds.read_flood_map(single_footprint)
    # convert hazard data to epsg 3857
    if '.shp' or '.geojson' in str(hazard_map):
        hazard_map=gpd.read_file(hazard_map).to_crs(3857)[['w_depth_l','w_depth_u','geometry']] #take only necessary columns (lower and upper bounds of water depth and geometry)
    else:
        hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
    
    hazard_map['geometry'] = hazard_map['geometry'].make_valid() if not hazard_map['geometry'].is_valid.all() else hazard_map['geometry']
    intersected_areas=overlay_hazard_adaptation_areas(hazard_map,adaptation_areas)
    overlay_adaptation_areas = pd.DataFrame(intersected_areas.T,columns=['adaptation_area','hazard_point'])
    geom_dict_aa = adaptation_areas['geometry'].to_dict()

    adaptations_cost_dict={}
    for adaptation_area in tqdm(overlay_adaptation_areas.groupby('adaptation_area'), total=len(overlay_adaptation_areas.adaptation_area.unique())): #adapted from Koks
        adapt_segment_geom = geom_dict_aa[adaptation_area[0]]

        adaptations_cost_dict = get_cost_per_area(adaptation_area,hazard_numpified_list[-1],adapt_segment_geom, adaptation_unit_cost)
    return adaptations_cost_dict 