from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.validation import make_valid
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


