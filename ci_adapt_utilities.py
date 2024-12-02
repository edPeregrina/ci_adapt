import numpy as np
import pandas as pd
import pickle
import geopandas as gpd
from tqdm.notebook import tqdm
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
import ast



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
        hazard_map=gpd.read_file(hazard_map).to_crs(3857)[['w_depth_l','w_depth_u', 'flood_area','geometry']] #take only necessary columns (lower and upper bounds of water depth and geometry)
    else:
        hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)

    hazard_map = hazard_map[hazard_map['flood_area'] == 1] # filter out flood protected areas
    hazard_map = hazard_map.drop('flood_area', axis=1)
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

def run_damage_reduction_by_asset(assets, geom_dict, overlay_assets, hazard_numpified_list, collect_inb_bl, changed_assets, hazard_intensity, fragility_values, maxdams_filt, map_rp_spec = None, asset_options=None, rp_spec_priority = None, reporting=False, adaptation_unit_cost=22500):
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
    # unchanged_assets = []

    if asset_options is not None and rp_spec_priority is not None and map_rp_spec is not None:
        skip_bridge, skip_tunnel = process_asset_options(asset_options, map_rp_spec, rp_spec_priority)
    else:
        skip_bridge=False
        skip_tunnel=False

    # interate over all unique assets and skip those that are not changed
    for asset in overlay_assets.groupby('asset'): #asset is a tuple where asset[0] is the asset index or identifier and asset[1] is the asset-specific information
        if asset[0] not in changed_assets.index:
            # unchanged_assets.append(asset[0])
            collect_inb_adapt[asset[0]] = collect_inb_bl[asset[0]]
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
    #sort return period categories from high to low
    return_period_dict = {k: v for k, v in sorted(return_period_dict.items(), key=lambda item: item[1])}
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


def calculate_new_paths(graph_v, shortest_paths, disrupted_edges, demand_reduction_dict=dict()):
    """
    Calculates new shortest paths in a graph after removing disrupted edges.

    Args:
        graph_v (Graph): Graph representing the infrastructure network.
        shortest_paths (dict): Dictionary of shortest paths.
        disrupted_edges (list): List of disrupted edges.
        demand_reduction_dict (dict, optional): Dictionary of demand reductions by origin/destination. Defaults to dict().

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
            demand_reduction_factor=demand_reduction_dict[(origin,destination)] if (origin,destination) in demand_reduction_dict.keys() else 0
            try:
                disrupted_shortest_paths[(origin,destination)] = (nx.shortest_path(graph_v_disrupted, origin, destination, weight='weight'), demand*(1-demand_reduction_factor))
            except nx.NetworkXNoPath:
                print(f'No path between {origin} and {destination}. Cannot ship by train.')
                disrupted_shortest_paths[(origin,destination)] = (None, demand*(1-demand_reduction_factor))
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

def create_virtual_graph(graph, reporting=False):
    """
    Creates a virtual graph with dummy nodes and edges to simulate maximum capacities and weights.
    Adapted from code by Asgarpour, S.

    Args:
        graph (Graph): Graph representing the infrastructure network.

    Returns:
        Graph: Virtual graph with dummy nodes and edges.
    """
    print('Creating virtual graph...')	
    max_weight_graph = max(attr['weight'] for _, _, attr in graph.edges(data=True))
    
    max_capacity_graph = 1#max(attr['capacity'] for _, _, attr in graph.edges(data=True))
    
    if reporting:
        print('Max weight: '+str(max_weight_graph))
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

def find_shortest_paths_capacities(graph, route_data, simplified=False):
    """
    Finds the shortest paths between terminal nodes in a graph based on route data and assigns capacities to edges.

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

def run_direct_damage_reduction_by_hazmap(data_path, config_file, assets, geom_dict, overlay_assets, hazard_numpified_list, collect_inb_bl, adapted_assets, map_rp_spec=None, asset_options=None, rp_spec_priority = None, reporting=False, adaptation_unit_cost=22500):
    """
    Runs direct damage reduction analysis for a hazard map, calculating damages and adaptation costs.

    Args:
        data_path (Path): Path to data storage.
        config_file (str): Path to configuration file.
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
    # data_path = Path(pathlib.Path.home().parts[0]) / 'Data'
    # Load configuration with ini file (created running config.py)
    # config_file=r'C:\repos\ci_adapt\config_ci_adapt.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    hazard_type = config.get('DEFAULT', 'hazard_type')
    infra_type = config.get('DEFAULT', 'infra_type')
    vulnerability_data = config.get('DEFAULT', 'vulnerability_data')
    infra_curves, maxdams = ds.read_vul_maxdam(data_path, hazard_type, infra_type)
    max_damage_tables = pd.read_excel(data_path / vulnerability_data / 'Table_D3_Costs_V1.1.0.xlsx',sheet_name='Cost_Database',index_col=[0])

    hazard_intensity = infra_curves['F8.1'].index.values
    fragility_values = (np.nan_to_num(infra_curves['F8.1'].values,nan=(np.nanmax(infra_curves['F8.1'].values)))).flatten()
    maxdams_filt=max_damage_tables[max_damage_tables['ID number']=='F8.1']['Amount']
    adaptation_run = run_damage_reduction_by_asset(assets, geom_dict, overlay_assets, hazard_numpified_list, collect_inb_bl, adapted_assets, hazard_intensity, fragility_values, maxdams_filt, 
                                                   map_rp_spec=map_rp_spec, asset_options=asset_options, rp_spec_priority = rp_spec_priority, reporting=reporting, adaptation_unit_cost=adaptation_unit_cost)

    return adaptation_run       

def run_indirect_damages_by_hazmap(adaptation_run, assets, hazard_map, overlay_assets, disrupted_edges, shortest_paths, graph_v, average_train_load_tons, average_train_cost_per_ton_km, average_road_cost_per_ton_km, demand_reduction_dict=dict(), reporting=False):
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
        demand_reduction_dict (dict, optional): Dictionary of demand reductions by origin/destination. Defaults to dict().
        reporting (bool, optional): Whether to print reporting information. Defaults to False.

    Returns:
        float: Economic impact of the disruptions.
    """
    # For a given hazard map overlay, find all the assets that are fully protected
    fully_protected_assets=[asset_id for asset_id, damages in adaptation_run[1].items() if damages[0]==0 and damages[1]==0]

    # For a given hazard map overlay, find all assets that are not exposed to flooding
    unexposed_assets=[asset_id for asset_id in assets.index if asset_id not in overlay_assets.asset.values]
    unexposed_osm_ids=assets.loc[assets.index.isin(unexposed_assets)]['osm_id'].values

    disrupted_edges_adapted = recalculate_disrupted_edges(graph_v, assets, disrupted_edges, fully_protected_assets, unexposed_osm_ids)
    # find the disrupted edges and paths under adapted conditions


    disrupted_shortest_paths_adapted=calculate_new_paths(graph_v, shortest_paths, disrupted_edges_adapted, demand_reduction_dict)

    if disrupted_shortest_paths_adapted == {}: # No disrupted paths, no economic impact
        if reporting==True:
            print(f'No shortest paths disrupted for {hazard_map}. No economic impact.')
        return 0

    impact=calculate_economic_impact_shortest_paths(hazard_map, graph_v, shortest_paths, disrupted_shortest_paths_adapted, average_train_load_tons, average_train_cost_per_ton_km, average_road_cost_per_ton_km)
    
    if reporting==True:
        print('-- Indirect losses --')
        print(f'{hazard_map}, {impact:,.2f}')
        print('disrupted_edges baseline: ', disrupted_edges)    
        print('disrupted_edges_adapted: ', disrupted_edges_adapted)

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
        current_adaptation = adapted_assets.loc[asset_id]['l1_rp_spec']
        adaptation_spec = affected_assets.loc[asset_id]['rp_spec']
        
        if adaptation_spec not in rp_spec_priority:
            print(f'Warning: Adaptation spec {adaptation_spec} not in rp_spec_priority')
            continue
        
        current_prio=rp_spec_priority.index(current_adaptation)
        adaptation_prio=rp_spec_priority.index(adaptation_spec)
        if adaptation_prio < current_prio:
            adapted_assets.loc[asset_id, 'l1_adaptation'] = affected_assets.loc[asset_id]['prot_area']
            adapted_assets.loc[asset_id, 'l1_rp_spec'] = adaptation_spec
            
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
    
    adaptation_cost = length_01*adaptation_unit_cost/detour_index
    
    print('Applying adaptation: new connection between assets with osm_id ', osm_id_pair)
    print('Level 3 adaptation')
    return graph_v, adaptation_cost

def add_l4_adaptation(graph_v, shortest_paths, adapted_route_area, demand_reduction=1.0):
    """
    Adds a level 4 adaptation by reducing demand on disrupted paths.

    Args:
        graph_v (Graph): Graph representing the infrastructure network.
        shortest_paths (dict): Dictionary of shortest paths.
        adapted_route_area (GeoDataFrame): GeoDataFrame of the area where demand is reduced.
        demand_reduction (float, optional): Demand reduction factor, with 1 full reduction and 0 no reduction. Defaults to 1.0.
    """
    # Find the shortest paths that are within the adapted route area
    # Make a list of unique nodes that are origin or destination
    od_nodes = []
    for (from_node, to_node), (path, demand) in shortest_paths.items():
        od_nodes.append(from_node)
        od_nodes.append(to_node)
    od_nodes = list(set(od_nodes))
    #create gdf_od_nodes with name of node and geometry of node
    gdf_od_nodes = gpd.GeoDataFrame(od_nodes, columns=['od_nodes'], geometry=[graph_v.nodes[node]['geometry'] for node in od_nodes], crs=3857)
    nodes_reduced_demand = gpd.overlay(gdf_od_nodes, adapted_route_area, how='intersection')
    nodes_reduced_demand_list = list(nodes_reduced_demand.od_nodes)
    # Reduce demand on the paths that are within the adapted route area
    demand_reduction_dict = {}
    for (from_node, to_node), (path, demand) in shortest_paths.items():
        if from_node in nodes_reduced_demand_list or to_node in nodes_reduced_demand_list:
            demand_reduction_dict[(from_node,to_node)] = demand_reduction

    print('Applying adaptation: reduced demand for routes: ', [key for key in demand_reduction_dict.keys()])
    print('Level 4 adaptation')

    return demand_reduction_dict


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
        float: Total adaptation cost for the basin.
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
    for adaptation_area in overlay_adaptation_areas.groupby('adaptation_area'): #adapted from Koks
        adapt_segment_geom = geom_dict_aa[adaptation_area[0]]
        # adaptations_cost_dict = get_cost_per_area(adaptation_area,hazard_numpified_list[-1],adapt_segment_geom, adaptation_unit_cost)
        adaptations_cost_dict[adaptation_area[0]] = get_cost_per_area(adaptation_area,hazard_numpified_list[-1],adapt_segment_geom, adaptation_unit_cost)
    adaptation_cost_basin = sum(adaptations_cost_dict.values())
    return adaptation_cost_basin 

# Define other functions (development)
def find_basin_lists(basins, regions):
    """
    Finds lists of basins for tributaries and full flood areas based on basin and region data.

    Args:
        basins (GeoDataFrame): GeoDataFrame containing basin data.
        regions (GeoDataFrame): GeoDataFrame containing region data.            
    Returns:        
        tuple: Lists of basins for tributaries and full flood areas.        

    """

    intersect_basins_regions = gpd.overlay(basins, regions, how='intersection')
    exclude_main_rivers=intersect_basins_regions.loc[intersect_basins_regions['ORDER']==1]

    basins_exclusion_list = [x for x in exclude_main_rivers['HYBAS_ID'].values]
    basin_list_tributaries = set([x for x in intersect_basins_regions['HYBAS_ID'].values if x not in basins_exclusion_list]) 
    basin_list_full_flood = set(intersect_basins_regions['HYBAS_ID'].values)
    return basin_list_tributaries, basin_list_full_flood


# Function definitions
def find_basin_lists(basins, regions):
    """
    Finds lists of basins for tributaries and full flood areas based on basin and region data.

    Args:
        basins (GeoDataFrame): GeoDataFrame containing basin data.
        regions (GeoDataFrame): GeoDataFrame containing region data.

    Returns:
        tuple: Lists of basins for tributaries and full flood areas.
    """ 

    intersect_basins_regions = gpd.overlay(basins, regions, how='intersection')
    exclude_main_rivers=intersect_basins_regions.loc[intersect_basins_regions['ORDER']==1]

    basins_exclusion_list = [x for x in exclude_main_rivers['HYBAS_ID'].values]
    basin_list_tributaries = set([x for x in intersect_basins_regions['HYBAS_ID'].values if x not in basins_exclusion_list]) 
    basin_list_full_flood = set(intersect_basins_regions['HYBAS_ID'].values)
    return basin_list_tributaries, basin_list_full_flood


def calculate_l1_costs(local_haz_path, interim_data_path, adapted_area, adaptation_unit_costs, adapted_assets):
    """
    Calculates level 1 adaptation costs for a specified area.

    Args:
        local_haz_path (Path): Path to hazard data.
        adapted_area (GeoDataFrame): GeoDataFrame of the adapted area.
        adaptation_unit_costs (dict): Dictionary of adaptation unit costs.
        adapted_assets (GeoDataFrame): GeoDataFrame of the adapted assets.

    Returns:
        dict: Cost of level 1 adaptation for each hazard map area.
    
    """

    hazard_data_list = ds.read_hazard_data(local_haz_path)
    if adapted_area is not None:
        adapted_area=adapted_area.explode(index_parts=True).reset_index(drop=True) #added index_parts=True for future compatibility, remove if behaving erratically
        adapted_area['geometry'] = adapted_area['geometry'].apply(lambda x: shapely.LineString(x.exterior) if isinstance(x, (shapely.Polygon, shapely.MultiPolygon)) else x)
        adapted_area['buffered'] = shapely.buffer(adapted_area.geometry.values,distance=1)    

        l1_adaptation_costs = {}
        for (adaptation_id, ad) in adapted_area.iterrows():
            if ad.adapt_level != 1:
                continue

            for single_footprint in hazard_data_list:
                hazard_map = single_footprint.parts[-1].split('.')[0]     
            
                haz_rp=hazard_map.split('_')[-3]
                if haz_rp != ad.rp_spec.upper():
                    continue
            
                overlay_assets = load_baseline_run(hazard_map, interim_data_path, only_overlay=True)
                if set(overlay_assets.asset.values).isdisjoint(adapted_assets.index):
                    continue
                if 'fwall' not in ad.prot_area:
                    continue
                try:
                    hazard_numpified_list = load_baseline_run(hazard_map, interim_data_path)[1]
                    adaptation_gdf=gpd.GeoDataFrame(adapted_area.iloc[[adaptation_id]])
                    if adaptation_id not in l1_adaptation_costs.keys():
                        l1_adaptation_costs[adaptation_id] = process_adap_dat(single_footprint, adaptation_gdf, hazard_numpified_list, adaptation_unit_cost=adaptation_unit_costs['fwall'])
                    else:
                        l1_adaptation_costs[adaptation_id] += process_adap_dat(single_footprint, adaptation_gdf, hazard_numpified_list, adaptation_unit_cost=adaptation_unit_costs['fwall'])                    
                    continue

                except Exception as e:
                    print(f'Error occurred in {hazard_map}: {str(e)}')
                    continue
        return l1_adaptation_costs

def apply_adaptations(adapted_area, assets, collect_output, interim_data_path, rp_spec_priority, adaptation_unit_costs, shortest_paths, graph_v, added_links, adapted_route_area):
    """
    Applies adaptations to assets in a hazard area.

    Args:
        adapted_area (GeoDataFrame): GeoDataFrame of the adapted area.
        assets (GeoDataFrame): GeoDataFrame of the assets.
        collect_output (dict): Dictionary of baseline damages by asset.
        interim_data_path (Path): Path to interim data storage.
        rp_spec_priority (list): List of return period priorities.
        adaptation_unit_costs (dict): Dictionary of adaptation unit costs.
        shortest_paths (dict): Dictionary of shortest paths between origins and destinations.
        graph_v (Graph): Graph representing the infrastructure network.
        added_links (list): List of added links.
        adapted_route_area (GeoDataFrame): GeoDataFrame of the adapted route area.

    Returns:
        tuple: adapted_assets (DataFrame), adaptations_df (DataFrame), demand_reduction_dict (dict), l3_adaptation_costs (dict)
    """

    adaptations_df=create_adaptation_df(adapted_area)
    if adapted_area is not None:
        assets_to_adapt = filter_assets_to_adapt(assets, adapted_area)
        adapted_assets = assets.loc[assets.index.isin(assets_to_adapt.index)].copy()
        adapted_assets = add_adaptation_columns(adapted_assets)

        for (adaptation_id, ad) in adapted_area.iterrows():
            affected_assets=assets_to_adapt.loc[assets_to_adapt['adaptation_id']==adaptation_id].copy()  
            rp_specs = set(affected_assets['rp_spec'])

            for hazard_map in collect_output.keys():
                haz_rp=hazard_map.split('_')[-3]
                if haz_rp not in rp_specs:
                    continue
                overlay_assets = load_baseline_run(hazard_map, interim_data_path, only_overlay=True)
                if set(overlay_assets.asset.values).isdisjoint(affected_assets.index):
                    continue
                else: 
                    overlay_assets, hazard_numpified_list = load_baseline_run(hazard_map, interim_data_path)
                    adapted_assets = apply_asset_adaptations_in_haz_area(adapted_assets, affected_assets, overlay_assets, hazard_numpified_list, rp_spec_priority)
    else:
        adapted_assets = assets.iloc[0:0].copy()
        adapted_assets = add_adaptation_columns(adapted_assets)
    l3_adaptation_costs = {}
    for i,osm_id_pair in enumerate(added_links):
        graph_v, l3_ad_cost = add_l3_adaptation(graph_v, osm_id_pair, adaptation_unit_cost=adaptation_unit_costs['bridge'])
        l3_adaptation_costs[osm_id_pair] = l3_ad_cost
        l3_ad_sum=[i, 'NA', 3, 'NA', l3_ad_cost]
        adaptations_df.loc[i] = l3_ad_sum

    if adapted_route_area is not None:
        demand_reduction_dict = add_l4_adaptation(graph_v, shortest_paths, adapted_route_area, demand_reduction=1.0)
        for i,(o,d) in enumerate(demand_reduction_dict.keys()):
            l4_ad_sum=[i, 'NA', 4, ((o,d),demand_reduction_dict[(o,d)]),0]
            adaptations_df.loc[i] = l4_ad_sum
    else:
        demand_reduction_dict = {}

    return adapted_assets, adaptations_df, demand_reduction_dict, l3_adaptation_costs

def run_adapted_damages(data_path, config_file, collect_output, disrupted_edges_by_basin, interim_data_path, assets, geom_dict, adapted_assets, adaptations_df, rp_spec_priority, adaptation_unit_costs, shortest_paths, graph_v, average_train_load_tons, average_train_cost_per_ton_km, average_road_cost_per_ton_km, demand_reduction_dict, reporting=False):
    """
    Runs direct and indirect damage calculation with adaptations.

    Args:
        data_path (Path): Path to data storage.
        config_file (Path): Path to the configuration file.
        collect_output (dict): Dictionary of baseline damages by hazard map.
        disrupted_edges_by_basin (dict): Dictionary of disrupted edges by basin.
        interim_data_path (Path): Path to interim data storage.
        assets (GeoDataFrame): GeoDataFrame of the assets.
        geom_dict (dict): Dictionary of asset geometries.
        adapted_assets (DataFrame): DataFrame of adapted assets.
        adaptations_df (DataFrame): DataFrame of adaptations.
        rp_spec_priority (list): List of return period priorities.
        adaptation_unit_costs (dict): Dictionary of adaptation unit costs.
        shortest_paths (dict): Dictionary of shortest paths between origins and destinations.
        graph_v (Graph): Graph representing the infrastructure network.
        average_train_load_tons (float): Average train load in tons.
        average_train_cost_per_ton_km (float): Average train cost per ton-kilometer.
        average_road_cost_per_ton_km (float): Average road cost per ton-kilometer.
        demand_reduction_dict (dict): Dictionary of demand reductions by origin/destination.
        reporting (bool, optional): Flag to enable reporting. Defaults to False.

    Returns:
        tuple: direct_damages_adapted (dict), indirect_damages_adapted (dict), adaptation_run_full (dict), l2_adaptation_costs (dict)
    
    """
    direct_damages_adapted = {}
    indirect_damages_adapted = {}
    l2_adaptation_costs = {}
    overlay_assets_lists = {'flood_DERP_RW_M':[], 'flood_DERP_RW_H':[], 'flood_DERP_RW_L':[]}
    adaptation_run_full = {'flood_DERP_RW_M':[{},{}], 'flood_DERP_RW_H':[{},{}], 'flood_DERP_RW_L':[{},{}]}
    for hazard_map in tqdm(collect_output.keys(), desc='Processing adapted damages by hazard map', total=len(collect_output.keys())):
        hm_full_id=hazard_map.split('_4326')[0]
        map_rp_spec = hazard_map.split('_')[-3]
        overlay_assets, hazard_numpified_list = load_baseline_run(hazard_map, interim_data_path)
        overlay_assets_lists[hm_full_id].extend(overlay_assets.asset.values.tolist())
        adaptation_run = run_direct_damage_reduction_by_hazmap(data_path, config_file, assets, geom_dict, overlay_assets, hazard_numpified_list, collect_output[hazard_map], adapted_assets, map_rp_spec=map_rp_spec, rp_spec_priority=rp_spec_priority, reporting=reporting, adaptation_unit_cost=adaptation_unit_costs['viaduct'])

        # Sum the values for the first dictionary in the tuple
        adaptation_run_full[hm_full_id][0] = {k: [adaptation_run_full[hm_full_id][0].get(k, [0, 0])[0] + adaptation_run[0].get(k, [0, 0])[0], 
                                adaptation_run_full[hm_full_id][0].get(k, [0, 0])[1] + adaptation_run[0].get(k, [0, 0])[1]] 
                            for k in set(adaptation_run_full[hm_full_id][0]) | set(adaptation_run[0])}

        # Sum the values for the second dictionary in the tuple
        adaptation_run_full[hm_full_id][1] = {k: [adaptation_run_full[hm_full_id][1].get(k, [0, 0])[0] + adaptation_run[1].get(k, [0, 0])[0], 
                                adaptation_run_full[hm_full_id][1].get(k, [0, 0])[1] + adaptation_run[1].get(k, [0, 0])[1]] 
                            for k in set(adaptation_run_full[hm_full_id][1]) | set(adaptation_run[1])}

        direct_damages_adapted[hazard_map]=adaptation_run[1]
        if adaptation_run[2] == {}:
            pass
        else:
            l2_adaptation_costs[hazard_map] = adaptation_run[2]
        if direct_damages_adapted[hazard_map]=={} and 3 not in adaptations_df.adapt_level.unique() and 4 not in adaptations_df.adapt_level.unique():
            indirect_damages_adapted[hazard_map]={}
            continue
        disrupted_edges = disrupted_edges_by_basin[hazard_map] if hazard_map in disrupted_edges_by_basin.keys() else []
        indirect_damages_adapted[hazard_map] = run_indirect_damages_by_hazmap(adaptation_run, assets, hazard_map, overlay_assets, disrupted_edges, shortest_paths, graph_v, average_train_load_tons, average_train_cost_per_ton_km, average_road_cost_per_ton_km, demand_reduction_dict)

    return direct_damages_adapted, indirect_damages_adapted, adaptation_run_full, l2_adaptation_costs, overlay_assets_lists

def calculate_indirect_dmgs_fullflood(full_flood_event, overlay_assets_lists, adaptation_run_full, assets, all_disrupted_edges, shortest_paths, graph_v, average_train_load_tons, average_train_cost_per_ton_km, average_road_cost_per_ton_km, demand_reduction_dict):
    """
    Calculates indirect damages for full flood events.

    Args:
        full_flood_event (dict): Dictionary of full flood events.
        overlay_assets_lists (dict): Dictionary of overlay assets for each full flood event.
        adaptation_run_full (dict): Dictionary of adaptation runs for all hazard maps.
        assets (GeoDataFrame): Asset data.
        all_disrupted_edges (dict): Dictionary of disrupted edges by hazard map.
        shortest_paths (dict): Dictionary of shortest paths between origins and destinations.
        graph_v (Graph): Graph representing the infrastructure network.
        average_train_load_tons (float): Average train load in tons.
        average_train_cost_per_ton_km (float): Average train cost per ton-kilometer.
        average_road_cost_per_ton_km (float): Average road cost per ton-kilometer.
        demand_reduction_dict (dict): Dictionary of demand reductions by origin/destination.

    Returns:
        dict: Indirect damages for full flood events.
    """
    indirect_damages_adapted_full={}
    for hazard_map in tqdm(full_flood_event.keys(), total=len(full_flood_event.keys()), desc='Processing full flood events'):
        overlay_assets_full_dict = {i:overlay_assets_lists[hazard_map][i] for i in range(len(overlay_assets_lists[hazard_map]))}
        overlay_assets_full=pd.DataFrame(overlay_assets_full_dict, index=['asset']).T
        indirect_damages_adapted_full[hazard_map] = run_indirect_damages_by_hazmap(adaptation_run_full[hazard_map], assets, hazard_map, overlay_assets_full, all_disrupted_edges[hazard_map], shortest_paths, graph_v, average_train_load_tons, average_train_cost_per_ton_km, average_road_cost_per_ton_km, demand_reduction_dict)

    return indirect_damages_adapted_full

def process_adaptation_costs(adaptation_cost_dict):
    """
    Process the adaptation costs dictionary to get the total adaptation costs for each adaptation.

    Args:
    - adaptation_cost_dict (dict): A dictionary of dictionaries representing adaptation costs at various levels. The structure is as follows:
        {adapt_id: {level: values of level depend on the level.
                    for l1: {assets: float/int}
                    for l2: {hazard maps: {asset: float/int}}
                    for l3: {added link tuple: float/int}
                    for l4: {added link tuple: float/int}
                    }
        }
        adapt_id (str): The identifier for the adaptation.
        level (dict): A dictionary representing the level-specific adaptation costs.
            - l1: A dictionary with assets as keys and their corresponding costs (float/int) as values.
            - l2: A dictionary with hazard maps as keys and their corresponding costs (float/int) as values.
            - l3: A dictionary with added link tuples as keys and their corresponding costs (float/int) as values.

    Returns:
    - adaptation_costs (dict): A dictionary with the total adaptation costs for each adaptation.
    """

    adaptation_costs = {}
    for adapt_id, levels in adaptation_cost_dict.items():
        adaptation_costs[adapt_id] = 0
        for level, costs in levels.items():
            if costs is None:
                continue
            if level == 'l1':
                adaptation_costs[adapt_id]+= sum(costs.values())/1e6
            if level == 'l2':
                adaptation_costs[adapt_id]+= sum([sum(costs.values()) for costs in costs.values()])/1e6
            if level == 'l3':
                adaptation_costs[adapt_id]+= sum(costs.values())/1e6

    return adaptation_costs

def find_basin_lists(basins, order=1):
    """
    Find the list of basins that are tributaries and the list of basins that are full flood.

    Args:
    - basins (GeoDataFrame): A GeoDataFrame of HYBAS basins.

    Returns:
    - basin_list_tributaries (set): A set of HYBAS IDs of basins that are tributaries.
    - basin_list (set): A set of all the HYBAS IDs of basins.
    - order (int): The order of the main river to exclude from the basin list.  Default is 1.
    """

    # intersect_basins_regions = gpd.overlay(basins, regions, how='intersection')
    exclude_main_rivers=basins.loc[basins['ORDER']==order]

    basins_exclusion_list = [x for x in exclude_main_rivers['HYBAS_ID'].values]
    basin_list_tributaries = set([x for x in basins['HYBAS_ID'].values if x not in basins_exclusion_list]) 
    basin_list = set(basins['HYBAS_ID'].values)
    return basin_list_tributaries, basin_list

def process_raw_adaptations_output(direct_damages_baseline_sum, direct_damages_adapted, event_impacts, indirect_damages_adapted, adaptations_df):
    """
    Process the raw output of the adaptation analysis to produce a dataframe of the total damages for each hazard map.

    Args:
    - direct_damages_baseline_sum (dict): A dictionary of dictionaries representing the direct damages for each hazard map in the baseline scenario. The structure is as follows:
        {hazard_map: {asset: (float, float)}}.
    - direct_damages_adapted (dict): A dictionary of dictionaries representing the direct damages for each hazard map in the adapted scenario. The structure is as follows: 
        {hazard_map: {asset: (float, float)}}.
    - event_impacts (dict): A dictionary of the indirect damages for each hazard map. The structure is as follows:
        {hazard_map: float}.
    - indirect_damages_adapted (dict): A dictionary of the indirect damages for each hazard map. The structure is as follows:
        {hazard_map: float}.
    - adaptations_df (DataFrame): A DataFrame of the adaptation costs. 

    Returns:
    - total_damages_adapted_df_mill (DataFrame): A DataFrame of the total damages for each hazard map in million euros.

    """	
    rp_priorities = ['H', 'M', 'L', 'Unknown']
    total_damages_adapted={}
    direct_damages_adapted_sum = {key: (sum(v[0] for v in direct_damages_adapted[key].values()), sum(v[1] for v in direct_damages_adapted[key].values())) for key in direct_damages_adapted}

    for hazard_map in direct_damages_adapted.keys():

        map_rp_spec = hazard_map.split('_')[-3]

        adap_costs=adaptations_df['adaptation_cost']
        summed_adaptation_costs = sum(adap_costs)

        # Direct damages
        summed_dd_bl_lower=direct_damages_baseline_sum[hazard_map][0]
        summed_dd_bl_upper=direct_damages_baseline_sum[hazard_map][1]

        summed_dd_ad_lower=direct_damages_adapted_sum[hazard_map][0]
        summed_dd_ad_upper=direct_damages_adapted_sum[hazard_map][1]
        
        # Indirect damages
        if hazard_map not in event_impacts.keys():
            print(f'{hazard_map} not in event_impacts')
            id_bl=0
            id_ad=0
            id_ad_cleaned=0
        else:
            id_bl=event_impacts[hazard_map]
            id_ad=indirect_damages_adapted[hazard_map]
            if id_ad == 99999999999999: 
                print(f'{hazard_map} has no indirect damages')
                break
            id_ad_cleaned = 0 if id_ad == 99999999999999 else id_ad

        total_damages_adapted[hazard_map]=(map_rp_spec, summed_adaptation_costs, (summed_dd_bl_lower, summed_dd_bl_upper), (summed_dd_ad_lower, summed_dd_ad_upper), id_bl, id_ad_cleaned)
        
    total_damages_adapted_df=pd.DataFrame(total_damages_adapted)
    total_damages_adapted_df=total_damages_adapted_df.T
    total_damages_adapted_df.columns=['return_period','summed_adaptation_costs', 'summed_dd_bl', 'summed_dd_ad', 'indirect damage baseline [€]', 'indirect damage adapted [€]']

    # Round and turn to million euros for reporting
    total_damages_adapted_df_mill=total_damages_adapted_df.copy()
    total_damages_adapted_df_mill['summed_adaptation_costs [M€]']=total_damages_adapted_df_mill['summed_adaptation_costs']/1e6
    total_damages_adapted_df_mill['summed_dd_bl [M€]']=total_damages_adapted_df_mill['summed_dd_bl'].apply(lambda x: (x[0]/1e6, x[1]/1e6))
    total_damages_adapted_df_mill['direct damage baseline lower [M€]']=total_damages_adapted_df_mill['summed_dd_bl'].apply(lambda x: x[0]/1e6)
    total_damages_adapted_df_mill['direct damage baseline upper [M€]']=total_damages_adapted_df_mill['summed_dd_bl'].apply(lambda x: x[1]/1e6)
    total_damages_adapted_df_mill['summed_dd_ad [M€]']=total_damages_adapted_df_mill['summed_dd_ad'].apply(lambda x: (x[0]/1e6, x[1]/1e6))
    total_damages_adapted_df_mill['direct damage adapted lower [M€]']=total_damages_adapted_df_mill['summed_dd_ad'].apply(lambda x: x[0]/1e6)
    total_damages_adapted_df_mill['direct damage adapted upper [M€]']=total_damages_adapted_df_mill['summed_dd_ad'].apply(lambda x: x[1]/1e6)
    total_damages_adapted_df_mill['indirect damage baseline [M€]']=total_damages_adapted_df_mill['indirect damage baseline [€]']/1e6
    total_damages_adapted_df_mill['indirect damage adapted [M€]']=total_damages_adapted_df_mill['indirect damage adapted [€]']/1e6
    total_damages_adapted_df_mill.drop(['summed_adaptation_costs','summed_dd_bl', 'summed_dd_ad', 'indirect damage baseline [€]', 'indirect damage adapted [€]'], axis=1, inplace=True)

    total_damages_adapted_df_mill['return_period'] = pd.Categorical(total_damages_adapted_df_mill['return_period'], 
                                                                    categories=rp_priorities, ordered=True)

    # Define the mapping dictionary based on the rp_spec_priority tuple
    rp_spec_priority_dict = {'L': 1,'M': 2,'H': 3, None: 0}

    # Sort by descending index of the last part of the name with return period in order of rp_priorities
    total_damages_adapted_df_mill['sort_index'] = total_damages_adapted_df_mill.index.str.split('_').str[-1].astype(int) + total_damages_adapted_df_mill['return_period'].map(rp_spec_priority_dict)
    total_damages_adapted_df_mill = total_damages_adapted_df_mill.sort_values(by='sort_index', ascending=False)
    total_damages_adapted_df_mill.drop('sort_index', axis=1, inplace=True)

    return total_damages_adapted_df_mill

def calculate_disruption_summary(disrupted_asset_ids_filt, clipped_assets, assets, save_to_csv=False, output_path=None):
    """
    Calculate the disruption for assets.

    Args:
    - disrupted_asset_ids_filt (dict): A dictionary of disrupted assets filtered by return period. The structure is as follows:
        {rp_def: [asset_ids]}.      
    - clipped_assets (GeoDataFrame): A GeoDataFrame of the clipped assets.
    - assets (GeoDataFrame): A GeoDataFrame of the assets for length to be calculated.
    - save_to_csv (bool): A boolean indicating whether to save the disruption summary to a csv file. Default is False.
    """	
    total_disrupted_assets = []
    total_disrupted_assets_byrp = {rp_def: 0 for rp_def in disrupted_asset_ids_filt.keys()}
    for rp_def, asset_ids in disrupted_asset_ids_filt.items():
        total_disrupted_assets.extend(asset_ids)
        total_disrupted_assets_byrp[rp_def] = len(asset_ids)
    total_disrupted_assets = list(set(total_disrupted_assets))

    total_disrupted_asset_length = 0
    for asset_id in total_disrupted_assets:
        total_disrupted_asset_length += assets.loc[asset_id].geometry.length

    total_network_length = 0
    for asset_id in clipped_assets.index:
        total_network_length += assets.loc[asset_id].geometry.length

    print(f'Total exposed assets: {len(total_disrupted_assets)}')   
    print(f'Total assets: {len(clipped_assets)}')
    print(f'Fraction of disrupted assets: {len(total_disrupted_assets) / len(clipped_assets):.2f}')
    print(f'Fraction of disrupted asset by length: {total_disrupted_asset_length / total_network_length:.2f}')

    # Save to csv
    if save_to_csv:
        if output_path is None:
            raise ValueError('data_path must be provided if save_to_csv is True')
        disruption_summary = pd.DataFrame({
            'Total exposed assets': [len(total_disrupted_assets)],
            'Total assets': [len(clipped_assets)],
            'Fraction of exposed assets': [len(total_disrupted_assets) / len(clipped_assets)],

            'Fraction of exposed asset (by length)': [total_disrupted_asset_length / total_network_length]
        })
        disruption_summary.to_csv(output_path, index=False)

def prep_adapted_basins_gdf(basins_gdf, eadD_ad_by_ts_basin_incf, eadIT_ad_by_ts_basin_incf, adapt_id='baseline', inc_f='mean', clipping_gdf=None): 
    """	
    Prepare the geodataframe of basins with EAD values for plotting

    Args:
    basins_gdf (gpd.GeoDataFrame): Geodataframe of basins
    eadD_ad_by_ts_basin_incf (dict): Dictionary of EAD values for adapted direct damages
    eadIT_ad_by_ts_basin_incf (dict): Dictionary of EAD values for adapted indirect damages
    adapt_id (str): Adaptation identifier
    inc_f (str): Increase factor to use for climate change
    clipping_gdf (gpd.GeoDataFrame): Geodataframe for clipping the output (default: None)
    """	
    basin_list = basins_gdf.HYBAS_ID.values.tolist()
    
    # Initialize columns with default values
    basins_gdf['Average EAD_D_ad_t0'] = 0.0
    basins_gdf['Average EAD_D_ad_t100'] = 0.0
    basins_gdf['EAD_ID_ad_t0'] = 0.0
    basins_gdf['EAD_ID_ad_t100'] = 0.0

    # Update columns with actual values where available
    for basin in basin_list:
        if basin in eadD_ad_by_ts_basin_incf[adapt_id][inc_f]:
            basins_gdf.loc[basins_gdf.HYBAS_ID == basin, 'Average EAD_D_ad_t0'] = eadD_ad_by_ts_basin_incf[adapt_id][inc_f][basin].values[0].mean()
            basins_gdf.loc[basins_gdf.HYBAS_ID == basin, 'Average EAD_D_ad_t100'] = eadD_ad_by_ts_basin_incf[adapt_id][inc_f][basin].values[-1].mean()
        if basin in eadIT_ad_by_ts_basin_incf[adapt_id][inc_f]:
            basins_gdf.loc[basins_gdf.HYBAS_ID == basin, 'EAD_ID_ad_t0'] = eadIT_ad_by_ts_basin_incf[adapt_id][inc_f][basin].values[0][0]
            basins_gdf.loc[basins_gdf.HYBAS_ID == basin, 'EAD_ID_ad_t100'] = eadIT_ad_by_ts_basin_incf[adapt_id][inc_f][basin].values[-1][0]

    adapted_basins_gdf = basins_gdf[['HYBAS_ID', 'geometry', 'Average EAD_D_ad_t0', 'Average EAD_D_ad_t100', 'EAD_ID_ad_t0', 'EAD_ID_ad_t100']]
    
    if clipping_gdf is not None:
        adapted_basins_gdf = gpd.clip(adapted_basins_gdf, clipping_gdf)

    return adapted_basins_gdf

def get_od_geoms_from_sps(shortest_paths, graph_r0):
    """	
    Get the origin-destination geometries from the shortest paths dictionary

    Args:
    shortest_paths (dict): dictionary of shortest paths
    graph_r0 (networkx graph): graph object

    Returns:
    list: list of origin-destination geometries
    """	
    o_geoms=[]
    d_geoms=[]
    for (o,d), (path, demand) in shortest_paths.items():
        o_geoms.append({'name': graph_r0.nodes[o]['name'], 'geometry': graph_r0.nodes[o]['geometry']})
        d_geoms.append({'name': graph_r0.nodes[d]['name'], 'geometry': graph_r0.nodes[d]['geometry']})

    # Join the two lists of dictionaries
    od_geoms = o_geoms + d_geoms
    # remove duplicates
    od_geoms = [g for i, g in enumerate(od_geoms) if g not in od_geoms[i+1:]]

    return od_geoms

def get_asset_ids_from_sps(shortest_paths, graph_r0):
    """	
    Get the asset ids from the shortest paths dictionary

    Args:
    shortest_paths (dict): dictionary of shortest paths
    graph_r0 (networkx graph): graph object

    Returns:
    dict: dictionary of asset ids, with the key being the origin-destination pair, and the value being a tuple of the asset ids and the demand.
    """	
    shortest_paths_assets={}
    for (o,d), (path, demand) in shortest_paths.items():
        od_assets_by_sp=[]
        for i in range(len(path)-1):
            edge=graph_r0.edges[path[i], path[i+1], 0]
            od_assets_by_sp.append(edge['osm_id'])
        shortest_paths_assets[(o,d)]=(od_assets_by_sp, demand)

    return shortest_paths_assets

def find_adapted_basin(eadD_ad_by_ts_basin_incf, eadIT_ad_by_ts_basin_incf, adapt_id='l1_trib'):
    # for each basin , find the difference between the eadD in baseline conditions and the eadD in adapted conditions and make a list for non-0 values. repeat for eadIT. put both lists together and return the list of basins with non-0 values
    adapted_basins_list = []
    for basin in eadD_ad_by_ts_basin_incf[adapt_id]['mean'].keys():
        try:    
            eadD_diff = eadD_ad_by_ts_basin_incf[adapt_id]['mean'][basin].values[0].mean() - eadD_ad_by_ts_basin_incf['baseline']['mean'][basin].values[0].mean()
            eadIT_diff = eadIT_ad_by_ts_basin_incf[adapt_id]['mean'][basin].values[0][0] - eadIT_ad_by_ts_basin_incf['baseline']['mean'][basin].values[0][0]
            if eadD_diff != 0 or eadIT_diff != 0:
                adapted_basins_list.append(basin)
        except Exception as e:
            pass
    adapted_basins_list=list(set(adapted_basins_list))

    return adapted_basins_list

def get_l3_gdf(added_links, graph_v):
    l3_geometries = {}
    if added_links != []:
        for u, v, k, attr in graph_v.edges(keys=True, data=True):
            if 'osm_id' not in attr:
                continue
            if 'l3_adaptation' in attr['osm_id']:
                # Ensure the geometry is a valid Shapely geometry object
                geometry = attr['geometry']
                if isinstance(geometry, list) and len(geometry) == 1 and isinstance(geometry[0], shapely.LineString):
                    geometry = geometry[0]
                if isinstance(geometry, shapely.LineString):
                    l3_geometries[(u, v)] = geometry
                else:
                    print(f"Invalid geometry for edge ({u}, {v}): {geometry}")

        gdf_l3_edges = gpd.GeoDataFrame.from_dict(l3_geometries, orient='index', columns=['geometry'], geometry='geometry', crs=3857)
        gdf_l3_edges.reset_index(inplace=True)

        gdf_l3_edges = gpd.GeoDataFrame(list(l3_geometries.items()), columns=['edge', 'geometry'], geometry='geometry', crs=3857).to_crs(4326)
        gdf_l3_edges['u']=gdf_l3_edges['edge'].apply(lambda x: x[0])
        gdf_l3_edges['v']=gdf_l3_edges['edge'].apply(lambda x: x[1])
        gdf_l3_edges=gdf_l3_edges.drop(columns=['edge'])

        return gdf_l3_edges
    else:
        return None

def get_l3_gdf(added_links, graph_v):
    l3_geometries = {}
    if added_links != []:
        for u, v, k, attr in graph_v.edges(keys=True, data=True):
            if 'osm_id' not in attr:
                continue
            if 'l3_adaptation' in attr['osm_id']:
                # Ensure the geometry is a valid Shapely geometry object
                geometry = attr['geometry']
                if isinstance(geometry, list) and len(geometry) == 1 and isinstance(geometry[0], shapely.LineString):
                    geometry = geometry[0]
                if isinstance(geometry, shapely.LineString):
                    l3_geometries[(u, v)] = geometry
                else:
                    print(f"Invalid geometry for edge ({u}, {v}): {geometry}")

        gdf_l3_edges = gpd.GeoDataFrame.from_dict(l3_geometries, orient='index', columns=['geometry'], geometry='geometry', crs=3857)
        gdf_l3_edges.reset_index(inplace=True)

        gdf_l3_edges = gpd.GeoDataFrame(list(l3_geometries.items()), columns=['edge', 'geometry'], geometry='geometry', crs=3857).to_crs(4326)
        gdf_l3_edges['u']=gdf_l3_edges['edge'].apply(lambda x: x[0])
        gdf_l3_edges['v']=gdf_l3_edges['edge'].apply(lambda x: x[1])
        gdf_l3_edges=gdf_l3_edges.drop(columns=['edge'])

        return gdf_l3_edges
    else:
        return None

def calculate_risk(probabilities, damages_lower, damages_upper):
    """
    Calculate the risk for each timestep based on the probabilities and damages.

    Args:
    - probabilities (DataFrame): A DataFrame of hazard exceedence probabilities for each timestep.
    - damages_lower (Series): A Series of lower bound damages for each return period.
    - damages_upper (Series): A Series of upper bound damages for each return period.

    Returns:
    - risk_l (list): A list of lower bound risk for each timestep.
    - risk_u (list): A list of upper bound risk for each timestep.
    """	 

    risk_l = []
    risk_u = []

    for ts in range(len(probabilities.iloc[0])):    
        risk_l_ts = []
        risk_u_ts = []

        for rp in range(len(probabilities)-1):
            d_rp = probabilities.iloc[rp][ts] - probabilities.iloc[rp + 1][ts]
            trap_damage_l = 0.5 * (damages_lower.iloc[rp] + damages_lower.iloc[rp + 1])
            trap_damage_u = 0.5 * (damages_upper.iloc[rp] + damages_upper.iloc[rp + 1])
            risk_l_ts.append(d_rp * trap_damage_l)
            risk_u_ts.append(d_rp * trap_damage_u)
            if d_rp < 0:
                print('Negative probability d_rp:', d_rp)
        
        # Add the portion of damages corresponding to the tails of the distribution
        d0_rp = probabilities.iloc[-1][ts]
        damage_l0 = max(damages_lower)
        damage_u0 = max(damages_upper)
        risk_l_ts.append(d0_rp * damage_l0)
        risk_u_ts.append(d0_rp * damage_u0)
        if d0_rp < 0:
            print('Negative probability d0_rp:', d0_rp)

        d_end_rp = (1/4) - probabilities.iloc[0][ts]
        damage_l_end = 0.5 * min(damages_lower)
        damage_u_end = 0.5 * min(damages_upper)
        risk_l_ts.append(d_end_rp * damage_l_end)
        risk_u_ts.append(d_end_rp * damage_u_end)
        if d_end_rp < 0:
            print('Negative probability d_end_rp:', d_end_rp)

        risk_l.append(sum(risk_l_ts))
        risk_u.append(sum(risk_u_ts))
    
    return risk_l, risk_u

def compare_outputs(collect_output, direct_damages_adapted, event_impacts, indirect_damages_adapted):
    changes_direct = 0
    changes_indirect = 0
    for key in collect_output.keys():
        no_change_direct = collect_output[key] == direct_damages_adapted[key]
        if no_change_direct == False:
            changes_direct += 1
        if key in event_impacts and key in indirect_damages_adapted:
            no_change_indirect = event_impacts[key] == indirect_damages_adapted[key]
            if no_change_indirect == False:
                changes_indirect += 1
        else:
            pass
    return print('Direct damage changes: ', changes_direct, '\nIndirect damage changes: ', changes_indirect)

def load_config(config_file):
    """
    Load the configuration file

    Args:
    config_file: str: path to the configuration file

    Returns:
    hazard_type: str: type of hazard
    infra_type: str: type of infrastructure
    country_code: str: country code
    country_name: str: country name
    hazard_data_subfolders: str: subfolders of the hazard data
    asset_data: str: path to the asset data
    vulnerability_data: str: path to the vulnerability data  
    """
    config = configparser.ConfigParser()
    config.read(config_file)
    hazard_type = config.get('DEFAULT', 'hazard_type')
    infra_type = config.get('DEFAULT', 'infra_type')
    country_code = config.get('DEFAULT', 'country_code')
    country_name = config.get('DEFAULT', 'country_name')
    hazard_data_subfolders = config.get('DEFAULT', 'hazard_data_subfolders')
    asset_data = config.get('DEFAULT', 'asset_data')
    vulnerability_data = config.get('DEFAULT', 'vulnerability_data')
    return hazard_type, infra_type, country_code, country_name, hazard_data_subfolders, asset_data, vulnerability_data

def startup_ci_adapt(data_path, config_file, interim_data_path=None):
    """
    Startup function for the ci_adapt model

    Args:
    data_path (Path): Path to the data folder
    interim_data_path (Path): Path to the interim data folder

    Returns:
    tuple: A tuple of the following data:   
        - assets (GeoDataFrame): A GeoDataFrame of the assets
        - geom_dict (dict): A dictionary of the geometries of the assets
        - miraca_colors (dict): A dictionary of colors for plotting
        - return_period_dict (dict): A dictionary of return periods
        - adaptation_unit_costs (dict): A dictionary of adaptation unit costs
        - rp_spec_priority (set): A set of return period priorities
        - average_road_cost_per_ton_km (float): Average road cost per ton per km
        - average_train_cost_per_ton_km (float): Average train cost per ton per km
        - average_train_load_tons (float): Average train load in tons
    """	
    # Load configuration with ini file (created running config.py)
    # config_file=r'C:\repos\ci_adapt\config_ci_adapt.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    # Set the display format for pandas
    pd.options.display.float_format = "{:,.3f}".format

    # Define paths
    asset_data = config.get('DEFAULT', 'asset_data')

    # Define colors for plotting
    color_string = config.get('DEFAULT', 'miraca_colors')
    miraca_colors = ast.literal_eval(color_string)

    # Define costs for different transport modes
    average_train_load_tons = (896+1344+2160+1344+896+896+1344+1512+896+390)/10 # in Tons per train. Source: Kennisinstituut voor Mobiliteitsbeleid. 2023. Cost Figures for Freight Transport – final report
    average_train_cost_per_ton_km = (0.014+0.018+0.047+0.045)/4 # in Euros per ton per km. Source: Kennisinstituut voor Mobiliteitsbeleid. 2023. Cost Figures for Freight Transport – final report
    average_road_cost_per_ton_km = (0.395+0.375+0.246+0.203+0.138+0.153+0.125+0.103+0.122+0.099)/10 # in Euros per ton per km. Source: Kennisinstituut voor Mobiliteitsbeleid. 2023. Cost Figures for Freight Transport – final report

    # Define dictionaries of return periods and adaptation unit costs
    return_period_dict = {'_H_': 10,'_M_': 100,'_L_': 200}
    adaptation_unit_costs = {'fwall': 7408, #considering floodwall in Germany
                            'viaduct': 36666, #considering viaduct cost
                            'bridge': 40102}  #considering bridge of 10m deck width
    rp_spec_priority = set_rp_priorities(return_period_dict)
    # Load data from baseline impact assessment
    assets_path = data_path / asset_data
    assets=preprocess_assets(assets_path)

    # Add buffer to assets to do area intersect and create dictionaries for quicker lookup
    # buffered_assets = ds.buffer_assets(assets)
    geom_dict = assets['geometry'].to_dict()

    print(f"{len(assets)} assets loaded.")

    return assets, geom_dict, miraca_colors, return_period_dict, adaptation_unit_costs, rp_spec_priority, average_road_cost_per_ton_km, average_train_cost_per_ton_km, average_train_load_tons

def load_baseline_impact_assessment(data_path):
    """	
    Load data from baseline impact assessment

    Args:
    data_path (Path): Path to the data folder

    Returns:
    tuple: A tuple of the following data:
        - shortest_paths (dict): dictionary of shortest paths
        - disrupted_edges_by_basin (dict): dictionary of disrupted edges by basin
        - graph_r0 (networkx graph): graph object
        - disrupted_shortest_paths (dict): dictionary of disrupted shortest paths
        - event_impacts (dict): dictionary of event impacts/losses
        - full_flood_event (dict): dictionary of full flood event losses
        - all_disrupted_edges (dict): dictionary of all disrupted edges
        - collect_output (dict): dictionary of collected output
    """	    
    shortest_paths = pickle.load(open(data_path / 'interim' / 'indirect_damages' / 'shortest_paths.pkl', 'rb'))
    disrupted_edges_by_basin = pickle.load(open(data_path / 'interim' / 'indirect_damages' / 'disrupted_edges_by_basin.pkl', 'rb'))
    graph_r0 = pickle.load(open(data_path / 'interim' / 'indirect_damages' / 'graph_0.pkl', 'rb'))
    disrupted_shortest_paths = pickle.load(open(data_path / 'interim' / 'indirect_damages' / 'disrupted_shortest_paths.pkl', 'rb'))
    event_impacts = pickle.load(open(data_path / 'interim' / 'indirect_damages' / 'event_impacts.pkl', 'rb'))
    full_flood_event=pickle.load(open(data_path / 'interim' / 'indirect_damages' / 'full_flood_event.pkl', 'rb'))
    all_disrupted_edges = pickle.load(open(data_path / 'interim' / 'indirect_damages' / 'all_disrupted_edges.pkl', 'rb'))
    collect_output = pickle.load(open(data_path / 'interim' / 'collected_flood_runs' / 'sample_collected_run.pkl', 'rb'))
    print('Loaded data from baseline impact assessment')
    return shortest_paths, disrupted_edges_by_basin, graph_r0, disrupted_shortest_paths, event_impacts, full_flood_event, all_disrupted_edges, collect_output

def load_basins_in_region(basins_path, regions_path, clipping=False):
    basins_gdf = gpd.read_file(basins_path)
    regions_gdf = gpd.read_file(regions_path)
    intersect_basins_regions = gpd.overlay(basins_gdf, regions_gdf, how='intersection')
    if clipping == True:
        intersect_basins_regions=gpd.clip(intersect_basins_regions, regions_gdf)
    
    return intersect_basins_regions
def load_adaptation_impacts(adaptation_id, data_path):
    """	
    Load data impact data for an adaptation id.

    Args:
    - adaptation_id (str): The identifier for the adaptation.
    - data_path (Path): Path to the data folder

    Returns:
    tuple: A tuple of the following data:
        - direct_damages_adapted (dict): A dictionary of dictionaries representing the direct damages for each tributary hazard map in the adapted scenario.
        - indirect_damages_adapted (dict): A dictionary of the indirect damages for each tributary hazard map in the adapted scenario.
        - indirect_damages_adapted_full (dict): A dictionary of the indirect damages for each full flood hazard map in the adapted scenario.
        - adapted_assets (dict): A dictionary of adapted assets.
        - adaptation_costs (dict): A dictionary of adaptation costs.
        - adaptations_df (DataFrame): A DataFrame of the adaptation.
    """	
    direct_damages_adapted_path = data_path / 'output' / f'adapted_direct_damages_{adaptation_id}.pkl'
    indirect_damages_adapted_path = data_path / 'output' / f'adapted_indirect_damages_{adaptation_id}.pkl'
    indirect_damages_adapted_full_path = data_path / 'output' / f'adapted_indirect_damages_full_{adaptation_id}.pkl'
    adapted_assets_path = data_path / 'output' / f'adapted_assets_{adaptation_id}.pkl'
    adaptation_costs_path = data_path / 'output' / f'adaptation_costs_{adaptation_id}.pkl'
    adaptations_df_path = data_path / 'interim' / 'adaptations' / f'{adaptation_id}_adaptations.csv'

    direct_damages_adapted = pickle.load(open(direct_damages_adapted_path, 'rb'))
    indirect_damages_adapted = pickle.load(open(indirect_damages_adapted_path, 'rb'))
    indirect_damages_adapted_full = pickle.load(open(indirect_damages_adapted_full_path, 'rb'))
    adapted_assets = pickle.load(open(adapted_assets_path, 'rb'))
    adaptation_costs = pickle.load(open(adaptation_costs_path, 'rb'))
    adaptations_df = pd.read_csv(adaptations_df_path)

    return direct_damages_adapted, indirect_damages_adapted, indirect_damages_adapted_full, adapted_assets, adaptation_costs, adaptations_df


def compile_direct_risk(inc_f, return_periods, basins_list, collect_output, total_damages_adapted_df_mill):
    """
    Compile the direct risk for each basin.

    Args:
    - inc_f (str): The increase factor to use for climate change.
    - return_periods (dict): A dictionary of return periods.
    - basins_list (list): A list of basins.
    - collect_output (dict): A dictionary of collected damage output.
    - total_damages_adapted_df_mill (DataFrame): A DataFrame of total damages including adaptations.

    Returns:
    - ead_y0_dd_all (numpy ndarray): An array of the expected annual direct damages at timestep 0 for all basins (upper and lower bounds).
    - ead_y100_dd_all (numpy ndarray): An array of the expected annual direct damages at timestep 100 for all basins (upper and lower bounds).
    - total_dd_all (numpy ndarray): An array of the total direct damages for all basins over all timesteps.
    - eadD_by_ts_by_basin (dict): A dictionary of the expected annual direct damages for each basin for each timestep. 
    """
    eadD_by_ts_by_basin_incf = {}
    eadD_by_ts_by_basin_incf[inc_f] = {}
    basin_dict = {}
    for basin in basins_list:
        basin_dict[basin] = {}
        for key in collect_output.keys():
            if str(basin) in key:

                basin_dict[basin][key.split('_RW')[-1][0:3]] = total_damages_adapted_df_mill.loc[key]['summed_dd_ad [M€]']
        
    aggregated_df_by_basin = {}
    eadD_by_ts_by_basin = {}
    for basin, damage_dict in basin_dict.items():
        if len(damage_dict) == 0:
            continue
        aggregated_df_by_basin[basin] = pd.DataFrame.from_dict(damage_dict, orient='index', columns=['Total Damage Lower Bound', 'Total Damage Upper Bound'])
        aggregated_df_by_basin[basin]['Return Period'] = [return_periods[index] for index in aggregated_df_by_basin[basin].index]
        aggregated_df_by_basin[basin] = aggregated_df_by_basin[basin].sort_values('Return Period', ascending=True)
        aggregated_df_by_basin[basin]['Probability'] = [[1 / x for x in i] for i in aggregated_df_by_basin[basin]['Return Period']]
        probabilities = aggregated_df_by_basin[basin]['Probability']
        risk_l, risk_u = calculate_risk(probabilities, aggregated_df_by_basin[basin]['Total Damage Lower Bound'], aggregated_df_by_basin[basin]['Total Damage Upper Bound'])
        eadD_by_ts_by_basin[basin] = pd.DataFrame(list(zip(risk_l, risk_u)), columns=['Total Damage Lower Bound', 'Total Damage Upper Bound'])
        eadD_by_ts_by_basin_incf[inc_f][basin] = eadD_by_ts_by_basin[basin]
    ead_y0_dd_all = sum([eadD_by_ts_by_basin[basin].values[0] for basin in eadD_by_ts_by_basin])
    ead_y100_dd_all = sum([eadD_by_ts_by_basin[basin].values[-1] for basin in eadD_by_ts_by_basin])
    total_dd_all = sum([sum(eadD_by_ts_by_basin[basin].values) for basin in eadD_by_ts_by_basin])
    return ead_y0_dd_all, ead_y100_dd_all, total_dd_all, eadD_by_ts_by_basin

def compile_indirect_risk_tributaries(inc_f, return_periods, basins_list, basin_list_tributaries, collect_output, total_damages_adapted_df_mill):
    """
    Compile the indirect risk for each basin.

    Args:
    - inc_f (str): The increase factor to use for climate change.
    - return_periods (dict): A dictionary of return periods.
    - basins_list (list): A list of basins.
    - basin_list_tributaries (list): A list of tributaries.
    - collect_output (dict): A dictionary of collected damage output.
    - total_damages_adapted_df_mill (DataFrame): A DataFrame of total damages including adaptations.

    Returns:
    - ead_y0_id_all (numpy ndarray): An array of the expected annual indirect damages at timestep 0 for all basins (upper and lower bounds).
    - ead_y100_id_all (numpy ndarray): An array of the expected annual indirect damages at timestep 100 for all basins (upper and lower bounds).
    - total_id_all (numpy ndarray): An array of the total indirect damages for all basins over all timesteps.
    - eadIT_by_ts_by_basin (dict): A dictionary of the expected annual indirect damages for each basin for each timestep.
    """	
    eadIT_by_ts_by_basin_incf = {}
    eadIT_by_ts_by_basin_incf[inc_f] = {}
    basin_dict = {}
    for basin in basins_list:
        if not basin in basin_list_tributaries:
            continue
        basin_dict[basin] = {}
        for key in collect_output.keys():
            if not str(basin) in key: 
                continue
            basin_dict[basin][key.split('_RW')[-1][0:3]] = total_damages_adapted_df_mill.loc[key]['indirect damage adapted [M€]']
                
    for rp in return_periods.keys():
        for basin in basin_dict.keys():
            if rp not in basin_dict[basin].keys():
                basin_dict[basin][rp] = 0
    aggregated_df_by_basin = {}
    eadIT_by_ts_by_basin = {}
    for basin, damage_dict in basin_dict.items():
        if len(damage_dict) == 0: 
            continue
        aggregated_df_by_basin[basin] = pd.DataFrame.from_dict(damage_dict, orient='index', columns=['Total indirect damage'])
        aggregated_df_by_basin[basin]['Return Period'] = [return_periods[index] for index in aggregated_df_by_basin[basin].index]
        aggregated_df_by_basin[basin] = aggregated_df_by_basin[basin].sort_values('Return Period', ascending=True)
        aggregated_df_by_basin[basin]['Probability'] = [[1 / x for x in i] for i in aggregated_df_by_basin[basin]['Return Period']]
        probabilities = aggregated_df_by_basin[basin]['Probability']
        risk_l, risk_u = calculate_risk(probabilities, aggregated_df_by_basin[basin]['Total indirect damage'], aggregated_df_by_basin[basin]['Total indirect damage'])
        eadIT_by_ts_by_basin[basin] = pd.DataFrame(list(zip(risk_l, risk_u)), columns=['Total indirect damage Lower Bound', 'Total indirect damage Upper Bound'])
        eadIT_by_ts_by_basin_incf[inc_f][basin] = eadIT_by_ts_by_basin[basin]

    ead_y0_id_all = sum([eadIT_by_ts_by_basin[basin].values[0] for basin in eadIT_by_ts_by_basin])
    ead_y100_id_all = sum([eadIT_by_ts_by_basin[basin].values[-1] for basin in eadIT_by_ts_by_basin])
    total_id_all = sum([sum(eadIT_by_ts_by_basin[basin].values) for basin in eadIT_by_ts_by_basin])
    return ead_y0_id_all, ead_y100_id_all, total_id_all, eadIT_by_ts_by_basin

def compile_indirect_risk_full_flood(return_periods, indirect_damages_adapted_full):
    """
    Compile the indirect risk for the full flood scenario.

    Args:
    - return_periods (dict): A dictionary of return periods.
    - indirect_damages_adapted_full (dict): A dictionary of indirect damages for the full flood scenario.

    Returns:
    - ead_y0_id_full (float): The expected annual indirect damages at timestep 0 for the full flood scenario.
    - ead_y100_id_full (float): The expected annual indirect damages at timestep 100 for the full flood scenario.
    - total_id_full (float): The total indirect damages for the full flood scenario.
    """
    aggregated_df = pd.DataFrame.from_dict(indirect_damages_adapted_full, orient='index', columns=['Total indirect damage'])
    # aggregated_df = aggregated_df.sort_values('Total indirect damage', ascending=True)
    aggregated_df['Total indirect damage'] = aggregated_df['Total indirect damage'] / 1e6
    rp_index = aggregated_df.index.str.split('_').str[-1]
    aggregated_df['Return Period'] = [return_periods['_'+index+'_'] for index in rp_index]
    aggregated_df = aggregated_df.sort_values('Return Period', ascending=True)
    aggregated_df['Probability'] = [[1 / rp for rp in ts] for ts in aggregated_df['Return Period']]
    probabilities = pd.DataFrame([[1 / rp for rp in ts] for ts in aggregated_df['Return Period']])
    risk_l, risk_u = calculate_risk(probabilities, aggregated_df['Total indirect damage'], aggregated_df['Total indirect damage'])
    ead_y0_id_full = risk_l[0]
    ead_y100_id_full = risk_l[-1]
    total_id_full = sum(risk_l)

    return ead_y0_id_full, ead_y100_id_full, total_id_full

