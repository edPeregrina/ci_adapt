import pandas as pd
import shapely
import numpy as np
import xarray as xr

#define paths
# p = Path('..')
# data_path = Path(pathlib.Path.home().parts[0]) / 'Data'


def country_download(iso3):
    dl.get_country_geofabrik(iso3)
    data_loc = OSM_DATA_DIR.joinpath(f'{DICT_GEOFABRIK[iso3][1]}-latest.osm.pbf')
    return data_loc

def overlay_hazard_assets(df_ds,assets):
    """
    Overlay hazard assets on a dataframe of spatial geometries.
    Arguments:
        *df_ds*: GeoDataFrame containing the spatial geometries of the hazard data. 
        *assets*: GeoDataFrame containing the infrastructure assets.
    Returns:
        *geopandas.GeoSeries*: A GeoSeries containing the spatial geometries of df_ds that intersect with the infrastructure assets.
    """
    #overlay #nts - review STRtree
    hazard_tree = shapely.STRtree(df_ds.geometry.values)
    if (shapely.get_type_id(assets.iloc[0].geometry) == 3) | (shapely.get_type_id(assets.iloc[0].geometry) == 6): # id types 3 and 6 stand for polygon and multipolygon
        return  hazard_tree.query(assets.geometry,predicate='intersects')    
    else:
        return  hazard_tree.query(assets.buffered,predicate='intersects')

def buffer_assets(assets,buffer_size=0.00083):
    """
    Buffer spatial assets in a GeoDataFrame.
    Arguments:
        *assets*: GeoDataFrame containing spatial geometries to be buffered.
        *buffer_size* (float, optional): The distance by which to buffer the geometries. Default is 0.00083.
    Returns:
        *GeoDataFrame*: A new GeoDataFrame with an additional 'buffered' column containing the buffered geometries.
    """
    assets['buffered'] = shapely.buffer(assets.geometry.values,distance=buffer_size)
    return assets


def get_damage_per_asset(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdams, double_track_factor): #added double rail factor (0.5) as each rail tracks of double rail are separate assets
    """
    Calculate damage for a given asset based on hazard information.
    Arguments:
        *asset*: Tuple containing information about the asset. It includes:
            - Index or identifier of the asset (asset[0]).
            - Asset-specific information, including hazard points (asset[1]['hazard_point']).  
        *flood_numpified*: NumPy array representing flood hazard information.
        *asset_geom*: Shapely geometry representing the spatial coordinates of the asset.
        *curve*: Pandas DataFrame representing the curve for the asset type.
        *maxdam*: Maximum damage value. #maxdams list of maxdam
    Returns:
        *list*: A list containing the asset index or identifier and the calculated damage.
    """
    
    # find the exact hazard overlays:
    get_hazard_points = hazard_numpified[asset[1]['hazard_point'].values] 
    get_hazard_points[shapely.intersects(get_hazard_points[:,1],asset_geom)]

    # estimate damage
    if len(get_hazard_points) == 0: # no overlay of asset with hazard
        return 0
    
    else:
        if asset_geom.geom_type == 'LineString':
            overlay_meters = shapely.length(shapely.intersection(get_hazard_points[:,1],asset_geom)) # get the length of exposed meters per hazard cell
            return [np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_meters*maxdam_asset*double_track_factor) for maxdam_asset in maxdams] #return asset number, total damage for asset number (damage factor * meters * max. damage)
        elif asset_geom.geom_type in ['MultiPolygon','Polygon']:
            overlay_m2 = shapely.area(shapely.intersection(get_hazard_points[:,1],asset_geom))
            return [np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_m2*maxdam_asset) for maxdam_asset in maxdams]
        elif asset_geom.geom_type == 'Point':
            return [np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*maxdam_asset) for maxdam_asset in maxdams]

def read_hazard_data(data_path,hazard_type='fluvial',country='Germany',defended=False,subfolders=None):

    if hazard_type == 'fluvial' and defended == False:
        hazard_data = data_path 
        #return [file for file in hazard_data.iterdir() if file.is_file() and file.suffix == '.shp']
        return [file for file in hazard_data.iterdir() if file.is_file() and file.suffix == '.geojson']
        #return list(hazard_data.iterdir())
    if hazard_type == 'fluvial' and defended == True:
        hazard_data = data_path / 'Floods' / country / 'fluvial_defended' / subfolders
        #return [file for file in hazard_data.iterdir() if file.is_file() and file.suffix == '.shp']
        return [file for file in hazard_data.iterdir() if file.is_file() and file.suffix == '.geojson']
        #return list(hazard_data.iterdir())

    else:
        hazard_data = data_path / 'Floods' / 'Germany' / 'fluvial_undefended' / 'raw_subsample' / 'validated_geometries'# need to make country an input
        print('Warning! hazard not supported')
        return [file for file in hazard_data.iterdir() if file.is_file() and file.suffix == '.geojson']
     
def read_vul_maxdam(data_path,hazard_type,infra_type):

    vul_data = data_path / 'Vulnerability'

    if hazard_type in ['pluvial','fluvial']:  
        curves = pd.read_excel(vul_data / 'Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0.xlsx',sheet_name = 'F_Vuln_Depth',index_col=[0],header=[0,1,2,3,4])
    elif hazard_type == 'windstorm':
        curves = pd.read_excel(vul_data / 'Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0.xlsx',sheet_name = 'W_Vuln_V10m',index_col=[0],header=[0,1,2,3,4])

    infra_curves =  curves.loc[:, curves.columns.get_level_values('Infrastructure description').str.lower().str.contains(infra_type)]
    
    maxdam = pd.read_excel(vul_data / 'Table_D3_Costs_V1.1.0.xlsx',sheet_name='Cost_Database',index_col=[1])
    infra_descriptions=maxdam.index.get_level_values('Infrastructure description').str.lower().str.contains(infra_type)
    infra_maxdam = maxdam.loc[infra_descriptions,'Amount'].dropna()
    infra_maxdam = infra_maxdam[pd.to_numeric(infra_maxdam, errors='coerce').notnull()]
    
    return infra_curves,infra_maxdam


def read_flood_map(flood_map_path):

    # check if vector and return path, and vectorize if raster"
    if '.shp' or '.geojson' in str(flood_map_path):
        return flood_map_path
    
    else: print('Vectorizing...')

    flood_map = xr.open_dataset(flood_map_path, engine="rasterio")

    flood_map_vector = flood_map['band_data'].to_dataframe().reset_index() #transform to dataframe
    
    # remove data that will not be used
    flood_map_vector = flood_map_vector.loc[(flood_map_vector.band_data > 0) & (flood_map_vector.band_data < 100)]
    
    # create geometry values and drop lat lon columns
    flood_map_vector['geometry'] = [shapely.points(x) for x in list(zip(flood_map_vector['x'],flood_map_vector['y']))]
    flood_map_vector = flood_map_vector.drop(['x','y','band','spatial_ref'],axis=1)
    
    # drop all non values to reduce size
    flood_map_vector = flood_map_vector.loc[~flood_map_vector['band_data'].isna()].reset_index(drop=True)
    
    # and turn them into squares again:
    flood_map_vector.geometry= shapely.buffer(flood_map_vector.geometry,distance=0.00083/2,cap_style='square').values # distance should be made an input still!

    return flood_map_vector


#plot assets and hazard map WARNING: can crash with many/complex geometries
#TODO make nicer with lonboard?
def subplots_asset_hazard(assets,hazard_map):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(6, 6), sharex=True, sharey=True)

    # Plot assets on the first subplot
    assets.plot(ax=axs[0])
    axs[0].set_title('Assets')

    # Plot hazard_map on the second subplot
    hazard_map.plot(ax=axs[1])
    axs[1].set_title('Hazard Footprint')

    plt.tight_layout()
    plt.show()

