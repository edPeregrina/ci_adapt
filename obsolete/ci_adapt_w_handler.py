####
##WORK IN PROGRESS, DO NOT USE
#H=Handler(config_file='config_ci_adapt.ini')
###

import configparser
import datetime
import pickle
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
# from shapely.validation import make_valid # only needed to make invalid geometries valid
from shapely import length, intersects, intersection
from pathlib import Path
import pathlib
from direct_damages import damagescanner_rail_track as ds


class Handler:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        self.hazard_type = config.get('DEFAULT', 'hazard_type')
        self.infra_type = config.get('DEFAULT', 'infra_type')
        self.country_code = config.get('DEFAULT', 'country_code')
        self.country_name = config.get('DEFAULT', 'country_name')
        self.hazard_data_subfolders = config.get('DEFAULT', 'hazard_data_subfolders')
        self.asset_data = config.get('DEFAULT', 'asset_data')
        self.vulnerability_data = config.get('DEFAULT', 'vulnerability_data')
        self.data_path = Path(pathlib.Path.home().parts[0]) / 'Data'
        self.interim_data_path = self.data_path / 'interim' / 'collected_flood_runs'
        p = Path('..')


    def read_asset_data(self):
        self.assets = Assets(self.data_path / self.asset_data)
        print("Assets loaded.")

    def read_hazard_data(self):
        self.hazard_data_list = ds.read_hazard_data(self.data_path, self.hazard_type, country=self.country_name, subfolders=self.hazard_data_subfolders)
        print(f'Found {len(self.hazard_data_list)} hazard maps.')

    def read_vul_maxdam(self):
        self.curve_types = {'primary': ['F7.1', 'F7.2'],
                       'secondary': ['F7.3', 'F7.4'],
                       'rail': ['F8.1']}
        self.infra_curves, self.maxdams = ds.read_vul_maxdam(self.data_path, self.hazard_type, self.infra_type)
        self.max_damage_tables = pd.read_excel(self.data_path / self.vulnerability_data / 'Table_D3_Costs_V1.0.0.xlsx',sheet_name='Cost_Database',index_col=[0])
        print(f'Found matching infrastructure curves for {self.infra_type}')

    def run_direct_damage(self):
        collect_output={}
        for i, single_footprint in enumerate(self.hazard_data_list):
            hazard_name = single_footprint.parts[-1].split('.')[0]
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'{timestamp} - Reading hazard map {i+1} of {len(self.hazard_data_list)}: {hazard_name}')

            try:
                collect_output[hazard_name] = process_hazard_data(single_footprint, self.hazard_type, self.assets.assets, self.interim_data_path, self.infra_curves, self.max_damage_tables, self.curve_types, self.infra_type, self.assets.type_dict, self.assets.geom_dict)
            except Exception as e:
                print(f'Error occurred in {hazard_name}: {str(e)}')
                continue
        
        self.collect_output = collect_output
        # save the data to pickle files
        collect_output_path = f'{self.interim_data_path}/sample_collected_run.pkl'
        if len(collect_output)>0:
            with open(collect_output_path, 'wb') as f:
                pickle.dump(collect_output, f)
        else: print('No output collected')

    def calculate_ead(self, dynamic_RPs=False):
        # calculate the expected annual damages (EAD)
        summed_output = {}
        # iterate over the items in the collect_output dictionary
        for hazard_map, asset_dict in self.collect_output.items():
            # if the hazard_map and hazard_curve combination is not already in the summed_output dictionary, add it with the sum of the current lower and upper bounds
            if hazard_map not in summed_output:
                summed_output[hazard_map] = (sum(value[0] for value in asset_dict.values()), sum(value[1] for value in asset_dict.values()))
            # if the hazard_map and hazard_curve combination is already in the summed_output dictionary, add the sum of the current lower and upper bounds to the existing ones
            else:
                summed_output[hazard_map][0] += sum(value[0] for value in asset_dict.values())
                summed_output[hazard_map][1] += sum(value[1] for value in asset_dict.values())


        # initialize a new dictionary to hold the aggregated values
        aggregated_output = {'_H_': [0, 0], '_M_': [0, 0], '_L_': [0, 0]}

        # iterate over the items in the summed_output dictionary and group into return periods
        for hazard_map, (lower_bound, upper_bound) in summed_output.items():
            # determine the category of the hazard map
            if '_H_' in hazard_map:
                category = '_H_'
            elif '_M_' in hazard_map:
                category = '_M_'
            else:  # '_L_' in hazard_map
                category = '_L_'
            # add the lower and upper bounds to the appropriate category
            aggregated_output[category][0] += lower_bound
            aggregated_output[category][1] += upper_bound

        # create the DataFrame from the new dictionary
        aggregated_df = pd.DataFrame.from_dict(aggregated_output, orient='index', columns=['Total Damage Lower Bound', 'Total Damage Upper Bound'])

        """
        Return period definitions:
        _H_=10-25y 
        _M_=100y
        _L_=200y (or more, check report)
        """
        # define dictionary to relate water depth classes to water depths, specific per region, in this case Rheinland Palatinate is used
        return_period_dict = {}
        return_period_dict['DERP'] = { #TODO: make generic
            '_H_': 10,
            '_M_': 100,
            '_L_': 200
        }

        # add the return period column to aggregated_df
        aggregated_df['Return Period'] = [return_period_dict['DERP'][index] for index in aggregated_df.index]
        print(aggregated_df)

        # sort the DataFrame by return period
        aggregated_df = aggregated_df.sort_values('Return Period', ascending=True)

        # Calculate the probability of each return period
        aggregated_df['Probability'] = 1 / aggregated_df['Return Period']
        probabilities = aggregated_df['Probability']
        dmgs = []
        for i in range(len(probabilities)):
            try:
                ead_l = 0.5 * ((probabilities.iloc[i] - probabilities.iloc[i + 1]) * (
                            aggregated_df['Total Damage Lower Bound'].iloc[i] + aggregated_df['Total Damage Lower Bound'].iloc[
                        i + 1]))
                ead_u = 0.5 * ((probabilities.iloc[i] - probabilities.iloc[i + 1]) * (
                            aggregated_df['Total Damage Upper Bound'].iloc[i] + aggregated_df['Total Damage Upper Bound'].iloc[
                        i + 1]))
                dmgs.append((ead_l, ead_u))
            except:
                pass

        ead_lower = 0
        ead_upper = 0
        for (ead_l, ead_u) in dmgs:
            ead_lower += ead_l
            ead_upper += ead_u

        ead = (ead_lower, ead_upper)
        print(f'Baseline expected annual damages: {ead}')
        return ead
    



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



class Assets:
    def __init__(self, file_path):
        self.assets = gpd.read_file(file_path)
        self.assets = gpd.GeoDataFrame(self.assets).set_crs(4326).to_crs(3857)
        self.assets = self.assets.loc[self.assets.geometry.geom_type == 'LineString']
        self.assets = self.assets.rename(columns={'railway' : 'asset'})

        # Uncomment the following lines if you want to drop passenger lines and light rails
        #self.assets = self.assets.loc[~(self.assets['railway:traffic_mode'] == 'passenger')]
        #self.assets = self.assets.loc[~(self.assets['asset'] == 'light_rail')]

        # Uncomment the following lines if you want to drop bridges and tunnels
        #self.assets = self.assets.loc[~(self.assets['bridge'].isin(['yes']))]
        #self.assets = self.assets.loc[~(self.assets['tunnel'].isin(['yes']))]

        self.buffered_assets = ds.buffer_assets(self.assets)
        self.geom_dict = self.assets['geometry'].to_dict()
        self.type_dict = self.assets['asset'].to_dict()


H=Handler(config_file='config_ci_adapt.ini')
H.read_vul_maxdam()
H.read_hazard_data()
H.read_asset_data()
H.run_direct_damage()
pd.DataFrame.from_dict(H.collect_output).to_csv(H.interim_data_path / 'sample_collected_run.csv')
H.ead=H.calculate_ead()

# TESTING ADAPTATION
# Create a variable changed_assets
changed_asset_list = [105426, 110740, 118116] # testing with a few assets

with open(H.interim_data_path / 'overlay_assets_flood_DERP_RW_L_4326_2080411370.pkl', 'rb') as f:
    overlay_assets = pickle.load(f)
with open(H.interim_data_path / 'hazard_numpified_flood_DERP_RW_L_4326_2080411370.pkl', 'rb') as f:
    hazard_numpified_list = pickle.load(f)
    
max_intensity = []
for asset_id in changed_asset_list:
    max_intensity.append(retrieve_max_intensity_by_asset(asset_id, overlay_assets, hazard_numpified_list))

changed_assets = H.assets.assets.loc[H.assets.assets.index.isin(changed_asset_list)].copy() # testing with a few assets
# Add new columns fragility_mod and haz_mod
changed_assets['fragility_mod'] = 1 #[0.3, 0.5, 0.8] #fraction (1 = no reduction, 0 = invulnerable asset) DUMMY DATA FOR TESTING
changed_assets['haz_mod'] = [np.max(x)+1 for x in max_intensity] #meters (0 = no reduction in hazard intensity, 0.5 = 0.5 meter reduction in hazard intensity) DUMMY DATA FOR TESTING consider raising railway 0.5 meters


hazard_intensity = H.infra_curves['F8.1'].index.values
fragility_values = (np.nan_to_num(H.infra_curves['F8.1'].values,nan=(np.nanmax(H.infra_curves['F8.1'].values)))).flatten()
maxdams_filt=H.max_damage_tables[H.max_damage_tables['ID number']=='F8.1']['Amount']

adaptation_run = run_damage_reduction_by_asset(H.assets.geom_dict, overlay_assets, hazard_numpified_list, changed_assets, hazard_intensity, fragility_values, maxdams_filt)

#reporting
for asset_id, baseline_damages in adaptation_run[0].items():
    print(f'\nADAPTATION results for asset {asset_id}:')
    print(f'Baseline damages for asset {asset_id}: {baseline_damages}')
    print(f'Adapted damages for asset {asset_id}: {adaptation_run[1][asset_id]}')
    delta = tuple(adaptation_run[1][asset_id][i] - baseline_damages[i] for i in range(len(baseline_damages)))
    percent_change = tuple((100 * (delta[i] / baseline_damages[i])) for i in range(len(baseline_damages)))

    print(f'Change (Adapted-Baseline): {delta}, {percent_change}% change, at a cost of {adaptation_run[2][asset_id]}')



