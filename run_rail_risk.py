import geopandas as gpd
import pandas as pd
#import xarray as xr
import numpy as np 
#import shapely 
from shapely.validation import make_valid
from tqdm import tqdm
from pathlib import Path
import pathlib
import pickle
import datetime
from from_elco import damagescanner_rail_track as ds

### source:  country_infrastructure_hazard() function Elco.
# define the paths to the data
p = Path('..')
data_path = Path(pathlib.Path.home().parts[0]) / 'Data'
interim_data_path = data_path / 'interim' / 'collected_flood_runs'

# use pre-downloaded data as source
rail_track_file=data_path / 'Exposure' / 'raw_rail_track_study_area_Rhine_Alpine_DEU.geojson'
#rail_track_file=data_path / 'Exposure' / 'raw_rail_track_study_area_KOBLENZ_BONN.geojson'

# create geodataframe from asset data
assets = gpd.read_file(rail_track_file)

# convert assets to epsg3857, filter lines and rename column
assets = gpd.GeoDataFrame(assets).set_crs(4326).to_crs(3857)
assets = assets.loc[assets.geometry.geom_type == 'LineString']
assets = assets.rename(columns={'railway' : 'asset'})

# drop passenger lines and light rails
#[Q1 - Kees] can drop light rail/trams as they are not freight? Could underestimate damage
#assets = assets.loc[~(assets['railway:traffic_mode'] == 'passenger')]
#assets = assets.loc[~(assets['asset'] == 'light_rail')]

# dropping bridges and tunnels
#assets = assets.loc[~(assets['bridge'].isin(['yes']))]
#assets = assets.loc[~(assets['tunnel'].isin(['yes']))]

# add buffer to assets
buffered_assets=ds.buffer_assets(assets)
buffered_assets.head(n=2)

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

# read vulnerability and maxdam data:
curve_types = {'primary': ['F7.1', 'F7.2'],
                'secondary': ['F7.3', 'F7.4'],
                'rail': ['F8.1']}

infra_curves,maxdams = ds.read_vul_maxdam(data_path,hazard_type,infra_type)
print(f'Found matching infrastructure curves for {infra_type}')
vul_data = data_path / 'Vulnerability' # Integrate into read_vul_maxdam function to make neater
maxdamxx = pd.read_excel(vul_data / 'Table_D3_Costs_V1.0.0.xlsx',sheet_name='Cost_Database',index_col=[0])

# define the path for the pickled damage results and try to load them
collect_output_path = f'{interim_data_path}/sample_collected_run.pkl'
if Path(collect_output_path).is_file():
    print('Damage results found in pickle files will be loaded')
    with open(collect_output_path, 'rb') as f:
        collect_output = pickle.load(f)

# if no damages found, caluclate damages
else:
    # Initialize the dictionary to hold the damage results for all maps and all hazard curves
    collect_output = {}
    for i,single_footprint in enumerate(hazard_data_list):
        # Define the paths for the pickled files
        try:
            hazard_name = single_footprint.parts[-1].split('.')[0]
            hazard_numpified_path = f'{interim_data_path}/hazard_numpified_{hazard_name}_{infra_curve[0]}.pkl'
            overlay_path = f'{interim_data_path}/overlay_assets_{hazard_name}_{infra_curve[0]}.pkl'
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'{timestamp} - Reading hazard map {i+1} of {len(hazard_data_list)}: {hazard_name}')

            # Check if the pickle files exist, load pickle and move to next file in the list if they do
            if Path(hazard_numpified_path).is_file() and Path(overlay_path).is_file():
                print('Flood maps found in pickle files will be loaded')
                # If they do, load the data from the pickle files
                with open(hazard_numpified_path, 'rb') as f:
                    hazard_numpified_list = pickle.load(f)
                with open(overlay_path, 'rb') as f:
                    hazard_numpified_list = pickle.load(f)                
                continue

            else: pass

            # load hazard map
            if hazard_type in ['pluvial','fluvial']:
                hazard_map = ds.read_flood_map(single_footprint)
            else: 
                print(f'{hazard_type} not implemented yet')
                continue 
                
            # convert hazard data to epsg 3857
            if '.shp' or '.geojson' in str(hazard_map):
                hazard_map=gpd.read_file(hazard_map).to_crs(3857)[['w_depth_l','w_depth_u','geometry']]
            else:
                hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
                            
            # make any invalid geometries valid: #time and maybe move to utility
            # print('Verifying validity of hazard footprint geometries')
            # hazard_map.geometry = hazard_map.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1) 
        
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'{timestamp} - Coarse overlay of hazard map with assets...')
            
            # coarse overlay of hazard map with assets
            intersected_assets=ds.overlay_hazard_assets(hazard_map,assets)
            overlay_assets = pd.DataFrame(intersected_assets.T,columns=['asset','hazard_point'])


            #[Q2 - Kees] - should we use the upper or lower bound for flooding depth? 
            # convert dataframe to numpy array
            # considering upper and lower bounds
            hazard_numpified_u = hazard_map.drop('w_depth_l', axis=1).to_numpy() 
            hazard_numpified_l = hazard_map.drop('w_depth_u', axis=1).to_numpy()
            hazard_numpified_list=[hazard_numpified_l, hazard_numpified_u] 
            
            # iterate over the infrastructure curves
            for infra_curve in infra_curves:
                collect_inb_path=f'{interim_data_path}/collect_inb_{hazard_name}_{infra_curve[0]}.pkl'
                if Path(collect_inb_path).is_file():
                    print('Damage results found in pickle files will be loaded')
                    with open(collect_inb_path, 'rb') as f:
                        collect_inb = pickle.load(f)
                    continue                
                
                maxdams_filt=maxdamxx[maxdamxx['ID number']==infra_curve[0]]['Amount']
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
                        
                        # get damage per asset and save in ditionary of assets:damage tuples
                        #collect_inb[asset[0]] = get_damage_per_asset(asset,hazard_numpified_l,asset_geom,hazard_intensity,fragility_values,maxdams_filt)
                        collect_inb[asset[0]] = tuple(ds.get_damage_per_asset(asset,h_numpified,asset_geom,hazard_intensity,fragility_values,maxdams_filt)[0] for h_numpified in hazard_numpified_list)

                with open(collect_inb_path, 'wb') as f:
                    pickle.dump(collect_inb, f)
           
                
                # after cycling through all assets in the map, add in-between dictionary to output-collection dictionary
                collect_output[(hazard_name,infra_curve[0])] = collect_inb
        except Exception as e:
            print(f'Error occurred in {hazard_name}: {str(e)}')
            continue

    # Save the data to pickle files
    with open(collect_output_path, 'wb') as f:
        pickle.dump(collect_output, f)    

# Calculate the expected annual damages (EAD)

# Initialize a new dictionary to hold the summed values
summed_output = {}

# Iterate over the items in the collect_output dictionary
for (hazard_map, hazard_curve), asset_dict in collect_output.items():
    # If the hazard_map and hazard_curve combination is not already in the summed_output dictionary, add it with the sum of the current lower and upper bounds
    if (hazard_map, hazard_curve) not in summed_output:
        summed_output[(hazard_map, hazard_curve)] = (sum(value[0] for value in asset_dict.values()), sum(value[1] for value in asset_dict.values()))
    # If the hazard_map and hazard_curve combination is already in the summed_output dictionary, add the sum of the current lower and upper bounds to the existing ones
    else:
        summed_output[(hazard_map, hazard_curve)][0] += sum(value[0] for value in asset_dict.values())
        summed_output[(hazard_map, hazard_curve)][1] += sum(value[1] for value in asset_dict.values())

# Print the summed_output dictionary
print(summed_output)

summary_df=pd.DataFrame.from_dict(summed_output,orient='index',columns=['Total Damage Lower Bound','Total Damage Upper Bound'])




# Initialize a new dictionary to hold the aggregated values
aggregated_output = {'_H_': [0, 0], '_M_': [0, 0], '_L_': [0, 0]}

# Iterate over the items in the summed_output dictionary and group into return periods
for (hazard_map, hazard_curve), (lower_bound, upper_bound) in summed_output.items():

    # Determine the category of the hazard map
    if '_H_' in hazard_map:
        category = '_H_'
    elif '_M_' in hazard_map:
        category = '_M_'
    else:  # '_L_' in hazard_map
        category = '_L_'
    # Add the lower and upper bounds to the appropriate category
    aggregated_output[category][0] += lower_bound
    aggregated_output[category][1] += upper_bound

# Create the DataFrame from the new dictionary
aggregated_df = pd.DataFrame.from_dict(aggregated_output, orient='index', columns=['Total Damage Lower Bound', 'Total Damage Upper Bound'])
aggregated_df
"""
Return period definitions:
_H_=10-25y 
_M_=100y
_L_=200y (or more, check report)
"""
# Define dictionary to relate water depth classes to water depths
return_period_dict = {}
return_period_dict['DERP'] = {
    '_H_': 10,
    '_M_': 100,
    '_L_': 200
}

# Add the return period column to aggregated_df
aggregated_df['Return Period'] = [return_period_dict['DERP'][index] for index in aggregated_df.index]

# Print the updated aggregated_df
print(aggregated_df)

# Sort the DataFrame by return period
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








