from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.validation import make_valid
from shapely import geometry
from tqdm import tqdm



def validate_hazard_maps(source_path=Path,hazard_map_name=str):
    """
    This function loads a hazard map from a given path and checks for invalid geometries. 
    If invalid geometries are found, the function will attempt to fix them.
    The function will return a geopandas dataframe with the fixed geometries.
    """
    # load flood data and reproject
    #   set data input Path and flood map name
    #root_dir = Path ("C:/","Data","Floods","Germany", "raw_data", "SzenarioSelten") #specify the path to the folder holding the input data
    print('Source map: ',end='')
    print(source_path)

    try:
        assert source_path.is_file()
    except AssertionError:
        print(f"Error: File {source_path} does not exist.")
        # Handle the error here, such as raising an exception or returning an error message
    try:
        gdf = gpd.read_file(source_path).to_crs(4326)
    except:
        gdf = gpd.read_file(source_path)
        print(f'Warning could not reproject to EPSG:4326! current CRS is {gdf.crs}.')
    
    gdf.geometry = gdf.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)
    
    return gdf

def validate_basin_geometries(basins_path,basins_path_valid):
    # make basin geometries valid
    basins=gpd.read_file(basins_path)
    basins.geometry = basins.apply(lambda row: make_valid(row.geometry) if not row.geometry.is_valid else row.geometry, axis=1)
    basins.to_file(basins_path_valid)
    

def qgis_intersect_basins(input_path=str, overlay_path=str, output_path=str, qgis_path = 'C:/Users/peregrin/AppData/Local/anaconda3/envs/qgis_env/Library/bin/qgis.exe'
):
    import os
    from qgis.core import QgsApplication

    # Set the environment variables for QGIS
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(qgis_path, 'apps', 'Qt5', 'plugins', 'platforms')
    os.environ['PATH'] += os.pathsep + os.path.join(qgis_path, 'apps', 'qgis', 'bin')
    os.environ['PYTHONPATH'] += os.pathsep + os.path.join(qgis_path, 'apps', 'qgis', 'python')
    os.environ['LD_LIBRARY_PATH'] = os.path.join(qgis_path, 'apps', 'qgis', 'lib')

    # Initialize QGIS application
    QgsApplication.setPrefixPath(qgis_path, True)
    qgs = QgsApplication([], False)
    qgs.initQgis()

    from qgis.analysis import QgsNativeAlgorithms
    QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

    import processing
    # intersect basin polygons with flood map
    processing.run("native:intersection", {'INPUT':input_path,'OVERLAY':overlay_path,'INPUT_FIELDS':['flood_area','depth_class','w_depth_l', 'w_depth_m', 'w_depth_u'],'OVERLAY_FIELDS':['HYBAS_ID','NEXT_DOWN','NEXT_SINK','MAIN_BAS','DIST_SINK','DIST_MAIN','SUB_AREA','UP_AREA','PFAF_ID','ORDER'],'OVERLAY_FIELDS_PREFIX':'','OUTPUT':output_path,'GRID_SIZE':None})

    # exit QGIS application
    qgs.exitQgis()

def split_by_hybas_id(gdf, flood_map, output_map_dir):
    _flood_map = flood_map.split('\\')[-1].split('_hybas_')[0]
    # Iterate over unique hybas_ids
    for unique_hybas_id in tqdm(gdf['HYBAS_ID'].unique()):
        output_file=f'flood_{_flood_map}_{unique_hybas_id}.geojson'
        output_path=Path(output_map_dir) / output_file
        # Select rows with the current unique value
        subset = gdf[gdf['HYBAS_ID'] == unique_hybas_id]  
        # Export the subset as GeoJSON in the output directory
        subset.to_file(output_path, driver='GeoJSON')
      
def simplify_geometries(gdf, tolerance=0.000014):
    gdf.geometry = gdf.geometry.simplify(tolerance)
    return gdf