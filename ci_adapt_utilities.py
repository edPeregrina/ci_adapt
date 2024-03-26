import numpy as np
import pandas as pd
import pickle
import geopandas as gpd
from tqdm import tqdm
import datetime
from shapely import length, intersects, intersection
from direct_damages import damagescanner_rail_track as ds


def process_hazard_data(single_footprint, hazard_type, assets, interim_data_path, infra_curves, max_damage_tables, curve_types, infra_type, type_dict, geom_dict):
    hazard_name = single_footprint.parts[-1].split('.')[0]
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

    # coarse overlay of hazard map with assets
    intersected_assets=ds.overlay_hazard_assets(hazard_map,assets)
    overlay_assets = pd.DataFrame(intersected_assets.T,columns=['asset','hazard_point'])

    # convert dataframe to numpy array
    # considering upper and lower bounds
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
                # retrieve asset geometry and do fine overlay
                asset_geom = geom_dict[asset[0]]              
                # get damage per asset in a single hazard map as a dictionary of asset IDs:damage tuples
                collect_inb[asset[0], infra_curve[0]] = tuple(ds.get_damage_per_asset(asset,h_numpified,asset_geom,hazard_intensity,fragility_values,maxdams_filt)[0] for h_numpified in hazard_numpified_list)

    return collect_inb

def retrieve_max_intensity_by_asset(asset, overlay_assets, hazard_numpified_list):
    # retrieve the hazard points that intersect with the asset
    max_intensity = hazard_numpified_list[0][overlay_assets.loc[overlay_assets['asset'] == asset].hazard_point.values] 
    # get the hazard intensity values for the hazard points
    return max_intensity[:,0]

def run_damage_reduction_by_asset(geom_dict, overlay_assets, hazard_numpified_list, changed_assets, hazard_intensity, fragility_values, maxdams_filt):
    # initialize dictionaries to hold the intermediate results
    collect_inb_bl ={}
    collect_inb_adapt = {}
    adaptation_cost={}
    unchanged_assets = []

    # interate over all unique assets and skip those that are not changed
    for asset in overlay_assets.groupby('asset'): #asset is a tuple where asset[0] is the asset index or identifier and asset[1] is the asset-specific information
        if asset[0] not in changed_assets.index:
            unchanged_assets.append(asset[0])
            continue
        print(f'Damage reduction for asset: {asset[0]}')

        # retrieve asset geometry
        asset_geom = geom_dict[asset[0]]

        # calculate damages for the baseline conditions 
        collect_inb_bl[asset[0]] = tuple(ds.get_damage_per_asset(asset,h_numpified,asset_geom,hazard_intensity,fragility_values,maxdams_filt)[0] for h_numpified in hazard_numpified_list)
        
        # calculate damages for the adapted conditions
        h_mod=changed_assets.loc[asset[0]].haz_mod #hazard modifier  (between 0 and the maximum hazard intensity)
        hazard_numpified_list_mod = [np.array([[max(0.0, x[0] - h_mod), x[1]] for x in haz_numpified_bounds]) for haz_numpified_bounds in hazard_numpified_list]
        frag_mod=changed_assets.loc[asset[0]].fragility_mod #fragility modifier (between 0 and the maximum fragility value, usually 1)

        collect_inb_adapt[asset[0]] = tuple(ds.get_damage_per_asset(asset,h_numpified,asset_geom,hazard_intensity,fragility_values*frag_mod,maxdams_filt)[0] for h_numpified in hazard_numpified_list_mod)
        
        # calculate the adaptation cost
        get_hazard_points = hazard_numpified_list_mod[0][asset[1]['hazard_point'].values] 
        get_hazard_points[intersects(get_hazard_points[:,1],asset_geom)]

        if len(get_hazard_points) == 0: # no overlay of asset with hazard
            affected_asset_length=0
        else:
            if asset_geom.geom_type == 'LineString':
                affected_asset_length = length(intersection(get_hazard_points[:,1],asset_geom)) # get the length of exposed meters per hazard cell

        adaptation_cost[asset[0]]=np.sum(h_mod*affected_asset_length*56454) # calculate the adaptation cost in EUR #TODO: include cost per meter as a variable
    print(f'Assets with no change: {unchanged_assets}')
    return collect_inb_bl, collect_inb_adapt, adaptation_cost

def calculate_dynamic_return_periods(return_period_dict, num_years, increase_factor):
    years = np.linspace(0, num_years, num_years + 1)
    return_periods = {}
    for category, rp in return_period_dict.items():
        rp_new = rp / increase_factor[category]
        rps = np.interp(years, [0, num_years], [rp, rp_new])
        return_periods[category] = rps.tolist()

    return return_periods

def ead_by_ts_plot(ead_by_ts):
    import matplotlib.pyplot as plt
    plt.fill_between(ead_by_ts.index, ead_by_ts['Total Damage Lower Bound'], ead_by_ts['Total Damage Upper Bound'], alpha=0.3, color='red')
    plt.title('Expected Annual Damages (EAD) over time')
    plt.xlabel('Years from baseline')
    plt.ylabel('EAD (euros)')
    plt.legend(['Damage Bounds'], loc='upper left')
    plt.ylim(0)  # Set y-axis lower limit to 0
    plt.show()


# def pickle_overlay_hazard(overlay_assets, hazard_numpified, damage_curve='8.1'):
#     with open('overlay_assets.pkl', 'wb') as f:
#         pickle.dump(overlay_assets, f)
#     with open('numpified_hazard.pkl', 'wb') as f:
#         pickle.dump(hazard_numpified, f)
#     with open('damage_curve.pkl', 'wb') as f:
#         pickle.dump(damage_curve, f)


## Check if the pickle files exist, load pickle and move to next file in the list if they do
# if Path(hazard_numpified_path).is_file() and Path(overlay_path).is_file():
#     print('Flood maps found in pickle files will be loaded')
#     # If they do, load the data from the pickle files
#     with open(hazard_numpified_path, 'rb') as f:
#         hazard_numpified_list = pickle.load(f)
#     with open(overlay_path, 'rb') as f:
#         overlay_assets = pickle.load(f)                
#     continue

# else: pass





# import geopandas as gpd
# from shapely import Point

# class GSNetwork:
#     def __init__(self,gdf_sources,gdf_sinks,buffer=0) -> None:
#         self.gdf_sources=gdf_sources
#         self.gdf_sinks=gdf_sinks
#         self.bbox=create_bounding_box(gdf_sinks,gdf_sources,buffer)

#     def retrieve_demand_sinks(gdf_sources):

# def create_bounding_box(gdf1,gdf2, buffer=0): #TODO MOVE TO UTILITIES
#         min_x = min(gdf1.total_bounds[0], gdf2.total_bounds[0]) - buffer
#         min_y = min(gdf1.total_bounds[1], gdf2.total_bounds[1]) - buffer
#         max_x = max(gdf1.total_bounds[2], gdf2.total_bounds[2]) + buffer
#         max_y = max(gdf1.total_bounds[3], gdf2.total_bounds[3]) + buffer
#         return (min_x, min_y, max_x, max_y)


# # Example GeoDataFrame s
# data1 = {'geometry': [Point(10, 30), Point(20, 25), Point(15, 35), Point(25, 41)]}
# data2 = {'geometry': [Point(9, 30), Point(20, 25), Point(15, 35), Point(25, 40)]}
# gdf_sources = gpd.GeoDataFrame(data1, geometry='geometry')
# gdf_sinks = gpd.GeoDataFrame(data2, geometry='geometry')

# # Example usage
# gs_nw_bbox = GSNetwork(gdf_sources,gdf_sinks,buffer=5)
# print(gs_nw_bbox.bbox)




    
        
        



# def calculate_ead(years_dict):
#     ead_results = {}

#     for year, damages in years_dict.items():
#         # Create a DataFrame for the damages
#         aggregated_df = pd.DataFrame({
#             'Total Damage Lower Bound': damages,
#             'Total Damage Upper Bound': damages,
#             'Return Period': [10, 100, 200]  # Corresponding to '_H_', '_M_', '_L_'
#         })

#         # Sort the DataFrame by return period
#         aggregated_df = aggregated_df.sort_values('Return Period', ascending=True)

#         # Calculate the probability of each return period
#         aggregated_df['Probability'] = 1 / aggregated_df['Return Period']
#         probabilities = aggregated_df['Probability']

#         ead_lower = 0
#         ead_upper = 0
#         for i in range(len(probabilities) - 1):
#             ead_l = 0.5 * ((probabilities.iloc[i] - probabilities.iloc[i + 1]) * (
#                         aggregated_df['Total Damage Lower Bound'].iloc[i] + aggregated_df['Total Damage Lower Bound'].iloc[
#                     i + 1]))
#             ead_u = 0.5 * ((probabilities.iloc[i] - probabilities.iloc[i + 1]) * (
#                         aggregated_df['Total Damage Upper Bound'].iloc[i] + aggregated_df['Total Damage Upper Bound'].iloc[
#                     i + 1]))
#             ead_lower += ead_l
#             ead_upper += ead_u

#         ead_results[year] = (ead_lower, ead_upper)

#     return ead_results








                




#         gdf_sinks_demand=gdf_sinks #+input
#         return gdf_sinks_demand
    
#     def retrieve_supply_sources(gdf_sources):
#         gdf_sources_supply=gdf_sources #+input
#         return gdf_sources_supply

# """
# These are utility functions that can be migrated to external utility file
# """  
#     # def get_demand(self,date=None):

# gdf_sinks='a'
# gdf_sources='b'   
# gs_network=GSNetwork(gdf_sinks,gdf_sources)
# # class ClassName:
# #     <statement-1>
# #     .
# #     .
# #     .
# #     <statement-N>

# # class Complex:
# #     def __init__(self, realpart, imagpart):
# #         self.r = realpart
# #         self.i = imagpart

# # x = Complex(3.0, -4.5)
# # x.r, x.i
# # (3.0, -4.5)

# # def scope_test():
# #     def do_local():
# #         spam = "local spam"

# #     def do_nonlocal():
# #         nonlocal spam
# #         spam = "nonlocal spam"

# #     def do_global():
# #         global spam
# #         spam = "global spam"

# #     spam = "test spam"
# #     do_local()
# #     print("After local assignment:", spam)
# #     do_nonlocal()
# #     print("After nonlocal assignment:", spam)
# #     do_global()
# #     print("After global assignment:", spam)

# # scope_test()
# # print("In global scope:", spam)