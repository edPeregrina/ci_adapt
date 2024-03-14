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

## for large data import from OSM:
# import osm_flex.download as dl
# import osm_flex.extract as ex
# from osm_flex.simplify import remove_contained_points,remove_exact_duplicates
# from osm_flex.config import OSM_DATA_DIR,DICT_GEOFABRIK

## for visualisation
# from lonboard import viz
# from lonboard.colormap import apply_continuous_cmap
# from palettable.colorbrewer.sequential import Blues_9

### source:  country_infrastructure_hazard() function Elco.
from from_elco import damagescanner_rail_track as ds
# Get the current timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


#define paths
p = Path('..')
data_path = Path(pathlib.Path.home().parts[0]) / 'Data'

# use pre-downloaded data as source
rail_track_file=data_path / 'Exposure' / 'raw_rail_track_study_area_Rhine_Alpine_DEU.geojson'
#rail_track_file=data_path / 'Exposure' / 'raw_rail_track_study_area_KOBLENZ_BONN.geojson'
assets = gpd.read_file(rail_track_file)

# convert assets to epsg3857, filter lines and rename column
assets = gpd.GeoDataFrame(assets).set_crs(4326).to_crs(3857)
assets = assets.loc[assets.geometry.geom_type == 'LineString']
assets = assets.rename(columns={'railway' : 'asset'})

# dropping passenger lines and light rails
#[Q1 - Kees] can drop light rail/trams as they are not freight? Could underestimate damage
#assets = assets.loc[~(assets['railway:traffic_mode'] == 'passenger')]
#assets = assets.loc[~(assets['asset'] == 'light_rail')]

# dropping bridges and tunnels
#assets = assets.loc[~(assets['bridge'].isin(['yes']))]
#assets = assets.loc[~(assets['tunnel'].isin(['yes']))]

# adding buffer to assets
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

# Integrate into read_vul_maxdam function to make neater
vul_data = data_path / 'Vulnerability'
maxdamxx = pd.read_excel(vul_data / 'Table_D3_Costs_V1.0.0.xlsx',sheet_name='Cost_Database',index_col=[0])


collect_output = {}



# for single_footprint in tqdm(hazard_data_list,total=len(hazard_data_list)):
for i,single_footprint in enumerate(hazard_data_list):
    try:
        hazard_name = single_footprint.parts[-1].split('.')[0]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{timestamp} - Reading hazard map {i+1} of {len(hazard_data_list)}: {hazard_name}')

        # load hazard map
        if hazard_type in ['pluvial','fluvial']:
            hazard_map = ds.read_flood_map(single_footprint)
        else: 
            print(f'{hazard_type} not implemented yet') 
            exit   
        # convert hazard data to epsg 3857
        if '.shp' or '.geojson' in str(hazard_map):
            hazard_map=gpd.read_file(hazard_map).to_crs(3857)[['w_depth_l','w_depth_u','geometry']]
        else:
            hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
                        
        # make any invalid geometries valid: #time and maybe move to utility
        # print('Verifying validity of hazard footprint geometries')
        # hazard_map.geometry = hazard_map.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1) 
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{timestamp} - Overlaying hazard map with assets')
        intersected_assets=ds.overlay_hazard_assets(hazard_map,assets)

        # overlay assets:
        overlay_assets = pd.DataFrame(intersected_assets.T,columns=['asset','hazard_point'])

        #[Q2 - Kees] - should we use the upper or lower bound for flooding depth? 
        # convert dataframe to numpy array
        # considering upper and lower bounds
        hazard_numpified_u = hazard_map.drop('w_depth_l', axis=1).to_numpy() 
        hazard_numpified_l = hazard_map.drop('w_depth_u', axis=1).to_numpy()
        hazard_numpified_list=[hazard_numpified_l, hazard_numpified_u] 


        # maxdam_types = {} #TODO use or remove?

        for infra_curve in infra_curves:
            maxdams_filt=maxdamxx[maxdamxx['ID number']==infra_curve[0]]['Amount']
            if not infra_curve[0] in curve_types[infra_type]:continue
            # get curves
            curve = infra_curves[infra_curve[0]]
            hazard_intensity = curve.index.values
            fragility_values = (np.nan_to_num(curve.values,nan=(np.nanmax(curve.values)))).flatten()       
            collect_inb = {}
            for asset in tqdm(overlay_assets.groupby('asset'),total=len(overlay_assets.asset.unique())): #group asset items for different hazard points per asset and get total number of unique assets
                try:
                    asset_type = type_dict[asset[0]]
                except KeyError: 
                    print(f'Passed asset {asset[0]}')
                    continue
                if not infra_curve[0] in curve_types[asset_type]: 
                    collect_inb[asset[0]] = 0
                    print(f'Asset {asset[0]}: No vulnerability data found')

                if np.max(fragility_values) == 0:
                    collect_inb[asset[0]] = 0  
                    print(f'Asset {asset[0]}: Fragility = 0')
                else:
                    asset_geom = geom_dict[asset[0]]
                                    
                    #collect_inb[asset[0]] = get_damage_per_asset(asset,hazard_numpified_l,asset_geom,hazard_intensity,fragility_values,maxdams_filt)
                    collect_inb[asset[0]] = tuple(ds.get_damage_per_asset(asset,h_numpified,asset_geom,hazard_intensity,fragility_values,maxdams_filt)[0] for h_numpified in hazard_numpified_list)
            #print(collect_inb)
            collect_output = pd.DataFrame.from_dict(collect_inb, orient='index')
    except: 
        print(f'Error! in {hazard_map}')
        continue
#break # remove break after testing
        
#subplots_asset_hazard(assets,hazard_map)

# Initialize a new dictionary to hold the summed values
summed_output = {}

# Iterate over the items in the collect_output dictionary
for (hazard_map, asset), df in collect_output.items():
    # If the hazard_map is not already in the summed_output dictionary, add it with the sum of the current lower and upper bounds
    if hazard_map not in summed_output:
        summed_output[hazard_map] = [df[0].sum(), df[1].sum()]
    # If the hazard_map is already in the summed_output dictionary, add the sum of the current lower and upper bounds to the existing ones
    else:
        summed_output[hazard_map][0] += df[0].sum()
        summed_output[hazard_map][1] += df[1].sum()

# Create the DataFrame from the new dictionary
summary_df = pd.DataFrame.from_dict(summed_output, orient='index', columns=['Total Damage Lower Bound', 'Total Damage Upper Bound'])

# Initialize a new dictionary to hold the aggregated values
aggregated_output = {'_H_': [0, 0], '_M_': [0, 0], '_L_': [0, 0]}

# Iterate over the items in the summed_output dictionary
for hazard_map, (lower_bound, upper_bound) in summed_output.items():
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

# Calculate the expected annual damages (EAD)

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

# Save all the current active variables to a file
filename_out = r'C:\Data\output\flood_DERP_EAD.pkl'
with open(filename_out, 'wb') as file:
    pickle.dump(locals(), file)

# Load the local variables from the file
with open(filename_out, 'rb') as f:
    local_vars = pickle.load(f)
ead_lower=0
ead_upper=0
for (ead_l, ead_u) in dmgs:
    ead_lower+=ead_l
    ead_upper+=ead_u

ead=(ead_lower, ead_upper)
# Save all the current active variables to a file
filename_out= r'C:\Data\output\flood_DERP_EAD.pkl'
with open(filename_out, 'wb') as file:
    pickle.dump(locals(), file)


# Load the local variables from the file
with open(filename_out, 'rb') as f:
    local_vars = pickle.load(f)