import geopandas as gpd
import pandas as pd
import numpy as np
# from shapely.validation import make_valid # only needed to make invalid geometries valid
from shapely import length, intersects, intersection
from tqdm import tqdm
from pathlib import Path
import pathlib
import pickle
import datetime
from direct_damages import damagescanner_rail_track as ds

### source:  country_infrastructure_hazard() function Elco.
# define the paths to the data
p = Path('..')
data_path = Path(pathlib.Path.home().parts[0]) / 'Data'
interim_data_path = data_path / 'interim' / 'collected_flood_runs'

# use pre-downloaded data as source
rail_track_file=data_path / 'Exposure' / 'raw_rail_track_study_area_Rhine_Alpine_DEU.geojson'

# create geodataframe from asset data
assets = gpd.read_file(rail_track_file)

# convert assets to epsg3857, filter lines and rename column
assets = gpd.GeoDataFrame(assets).set_crs(4326).to_crs(3857)
assets = assets.loc[assets.geometry.geom_type == 'LineString']
assets = assets.rename(columns={'railway' : 'asset'})

# drop passenger lines and light rails
#[Q1 - Kees] can drop light rail/trams as they are not freight?
#assets = assets.loc[~(assets['railway:traffic_mode'] == 'passenger')]
#assets = assets.loc[~(assets['asset'] == 'light_rail')]

# dropping bridges and tunnels
#assets = assets.loc[~(assets['bridge'].isin(['yes']))]
#assets = assets.loc[~(assets['tunnel'].isin(['yes']))]

# add buffer to assets
buffered_assets=ds.buffer_assets(assets)

# create dicts for quicker lookup
geom_dict = assets['geometry'].to_dict()
type_dict = assets['asset'].to_dict()
print('Identified asset types:')
print(set(type_dict.values()))

# define data source path
hazard_type='fluvial'
infra_type='rail'
country_code='DEU' 
country_name='Germany' #get from ISO
hazard_data_subfolders="raw_subsample/validated_geometries"

# read hazard data
hazard_data_list = ds.read_hazard_data(data_path,hazard_type,country=country_name,subfolders=hazard_data_subfolders)
print(f'Found {len(hazard_data_list)} hazard maps.')

# read vulnerability and maxdam data, currently includes roads and rail:
curve_types = {'primary': ['F7.1', 'F7.2'],
                'secondary': ['F7.3', 'F7.4'],
                'rail': ['F8.1']}

infra_curves,maxdams = ds.read_vul_maxdam(data_path,hazard_type,infra_type)
print(f'Found matching infrastructure curves for {infra_type}')
vul_data = data_path / 'Vulnerability' #TODO: Integrate into read_vul_maxdam function to make neater
max_damage_tables = pd.read_excel(vul_data / 'Table_D3_Costs_V1.0.0.xlsx',sheet_name='Cost_Database',index_col=[0])

# Initialize the dictionary to hold the damage results for all maps and all hazard curves
collect_output = {}
for i,single_footprint in enumerate(hazard_data_list):
    # define the paths for the pickled files
    try:
        # load hazard data
        hazard_name = single_footprint.parts[-1].split('.')[0]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{timestamp} - Reading hazard map {i+1} of {len(hazard_data_list)}: {hazard_name}')

        # load hazard map
        if hazard_type in ['pluvial','fluvial']:
            hazard_map = ds.read_flood_map(single_footprint)
        else: 
            print(f'{hazard_type} not implemented yet')
            continue 
            
        # convert hazard data to epsg 3857
        if '.shp' or '.geojson' in str(hazard_map):
            hazard_map=gpd.read_file(hazard_map).to_crs(3857)[['w_depth_l','w_depth_u','geometry']] #take only necessary columns (lower and upper bounds of water depth and geometry)
        else:
            hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
                        
        # make any invalid geometries valid: #time and maybe move to utility or handle as exception
        # print('Verifying validity of hazard footprint geometries')
        # hazard_map.geometry = hazard_map.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1) 
    
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{timestamp} - Coarse overlay of hazard map with assets...')
        
        # coarse overlay of hazard map with assets
        intersected_assets=ds.overlay_hazard_assets(hazard_map,assets)
        overlay_assets = pd.DataFrame(intersected_assets.T,columns=['asset','hazard_point'])

        #[Q2 - Kees] - currently using upper or lower bound for flooding depth. other options?
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
                    # get damage per asset as a dictionary of asset IDs:damage tuples
                    collect_inb[asset[0]] = tuple(ds.get_damage_per_asset(asset,h_numpified,asset_geom,hazard_intensity,fragility_values,maxdams_filt)[0] for h_numpified in hazard_numpified_list)
            
            # save the data to pickle files #TODO: remove? adaptation module recalculates anyway
            collect_inb_path=f'{interim_data_path}/collect_inb_{hazard_name}_{infra_curve[0]}.pkl' 
            with open(collect_inb_path, 'wb') as f:
                pickle.dump(collect_inb, f)

            # after cycling through all assets in the map, add in-between dictionary to output-collection dictionary
            collect_output[(hazard_name,infra_curve[0])] = collect_inb

    except Exception as e:
        print(f'Error occurred in {hazard_name}: {str(e)}')
        continue

# save the data to pickle files
collect_output_path = f'{interim_data_path}/sample_collected_run.pkl'
with open(collect_output_path, 'wb') as f:
    pickle.dump(collect_output, f)    

# calculate the expected annual damages (EAD)
summed_output = {}

# iterate over the items in the collect_output dictionary
for (hazard_map, hazard_curve), asset_dict in collect_output.items():
    # if the hazard_map and hazard_curve combination is not already in the summed_output dictionary, add it with the sum of the current lower and upper bounds
    if (hazard_map, hazard_curve) not in summed_output:
        summed_output[(hazard_map, hazard_curve)] = (sum(value[0] for value in asset_dict.values()), sum(value[1] for value in asset_dict.values()))
    # if the hazard_map and hazard_curve combination is already in the summed_output dictionary, add the sum of the current lower and upper bounds to the existing ones
    else:
        summed_output[(hazard_map, hazard_curve)][0] += sum(value[0] for value in asset_dict.values())
        summed_output[(hazard_map, hazard_curve)][1] += sum(value[1] for value in asset_dict.values())

# print the summed_output dictionary
print(summed_output)
summary_df=pd.DataFrame.from_dict(summed_output,orient='index',columns=['Total Damage Lower Bound','Total Damage Upper Bound'])

# initialize a new dictionary to hold the aggregated values
aggregated_output = {'_H_': [0, 0], '_M_': [0, 0], '_L_': [0, 0]}

# iterate over the items in the summed_output dictionary and group into return periods
for (hazard_map, hazard_curve), (lower_bound, upper_bound) in summed_output.items():

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
return_period_dict['DERP'] = {
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
#TODO: aggregate and report EAD for each region instead of for all regions combined


# function to recalculate damages for assets with changed fragility and hazard modifiers
def run_damage_reduction_by_asset(geom_dict, overlay_assets, hazard_numpified_list, changed_assets, hazard_intensity, fragility_values, maxdams_filt):
    # initialize dictionaries to hold the intermediate results
    collect_inb_bl ={}
    collect_inb_adapt = {}
    adaptation_cost={}

    # interate over all unique assets and skip those that are not changed
    for asset in overlay_assets.groupby('asset'): #asset is a tuple where asset[0] is the asset index or identifier and asset[1] is the asset-specific information
        if asset[0] not in changed_assets.index:
            print(f'Asset not in changed_assets: {asset[0]}')
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
        
    return collect_inb_bl, collect_inb_adapt, adaptation_cost


# Create a variable changed_assets
changed_assets = assets.loc[assets.index.isin([7819, 46896, 46901])].copy() # testing with a few assets

# Add new columns fragility_mod and haz_mod
changed_assets['fragility_mod'] = [0.3, 0.5, 0.8] #fraction DUMMY DATA FOR TESTING
changed_assets['haz_mod'] = 0.5 # meters DUMMY DATA FOR TESTING consider raising railway 0.5 meters
adaptation_run = run_damage_reduction_by_asset(geom_dict, overlay_assets, hazard_numpified_list, changed_assets, hazard_intensity, fragility_values, maxdams_filt)

#reporting
for asset_id, baseline_damages in adaptation_run[0].items():
    print(f'\nADAPTATION results for asset {asset_id}:')
    print(f'Baseline damages for asset {asset_id}: {baseline_damages}')
    print(f'Adapted damages for asset {asset_id}: {adaptation_run[1][asset_id]}')
    delta = tuple(adaptation_run[1][asset_id][i] - baseline_damages[i] for i in range(len(baseline_damages)))
    percent_change = tuple((100 * (delta[i] / baseline_damages[i])) for i in range(len(baseline_damages)))

    print(f'Change (Adapted-Baseline): {delta}, {percent_change}% change, at a cost of {adaptation_run[2][asset_id]}')



