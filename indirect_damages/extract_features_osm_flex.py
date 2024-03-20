import sys
import os
from osm_flex.download import *
from osm_flex.extract import *
from osm_flex.config import *
import osm_flex.clip as cp
import pickle

sys.path.append(r'C:\repos')
sys.path.append(r'C:\repos\ra2ce')
sys.path.append(r'C:\repos\ra2ce_multi_network')
from ra2ce_multi_network.simplify_rail import *
from ra2ce_multi_network.deeper_extraction import filter_on_other_tags
import json

### Defining ini variables
root_folder = OSM_DATA_DIR.parent

## dump-related
iso3_code = "DEU"
region_code = "Europe"

dump_region = DICT_GEOFABRIK[iso3_code][1]
# dump_region = region_code.lower()

dump_folder = root_folder / "osm_pbf"

## Clipping-related
study_area_suffix = '_Rhine_Alpine_DEU'  # small case study area that works: '_ROTTERDAM_PORT'
clip_polygon_path = Path(
    rf'C:\Users\peregrin\osm\osm_bpf\polygon_Rhine_Alpine_DEU.geojson'
)
clip_output_name = f'study_area{study_area_suffix}'
study_area_dump_path = root_folder.joinpath('osm_bpf', f'{clip_output_name}.osm.pbf')

## Extraction-related
default_osm_keys = DICT_CIS_OSM['rail']['osm_keys']
extract_path = root_folder.joinpath('extracts')

# source: https://taginfo.openstreetmap.org/tags/railway=rail#combinations
# 'other_tags' key is a string chain of 'tags' => 'keys', where relavant information is stored. e.g., whether traffic mode is freight or mixed
rail_track_attributes = {
    'osm_keys': [
        'railway', 'name', 'gauge', 'electrified', 'voltage', 'bridge', 'maxspeed', 'service', 'tunnel', 'other_tags'
    ],
    'other_tags': ['"railway:traffic_mode"=>', '"usage"=>']
}

rail_track_osm_query = """railway='rail' or railway='light_rail'"""

# Loading rail track files if they exists, creating them if they do not
raw_rail_track_file = root_folder.joinpath(f'raw_rail_track_{clip_output_name}.geojson')
rail_track_file = root_folder.joinpath(f'rail_track_{clip_output_name}.geojson')

try:
    assert raw_rail_track_file.is_file()
    fn=str(raw_rail_track_file).split('\\')[-1] 
    # load gdf from saved file
    raw_rail_track_gdf=gpd.read_file(raw_rail_track_file)
    print(f'File {fn} found and loaded')
except AssertionError as e:
    print('File not found, extracting data')
    raw_rail_track_gdf = extract(osm_path=study_area_dump_path, geo_type='lines',
                             osm_keys=rail_track_attributes['osm_keys'], osm_query=rail_track_osm_query)
    raw_rail_track_gdf.to_file(raw_rail_track_file, driver='GeoJSON')


try:
    assert rail_track_file.is_file() 
    fn=str(rail_track_file).split('\\')[-1]
    # load gdf from saved file
    rail_track_gdf=gpd.read_file(rail_track_file)
    print(f'File {fn} found and loaded')
except AssertionError as e:
    print('File not found, processing')
    rail_track_gdf = filter_on_other_tags(
    attributes=rail_track_attributes, other_tags_keys=rail_track_attributes['other_tags'], gdf=raw_rail_track_gdf
)
    rail_track_gdf.to_file(rail_track_file, driver='GeoJSON')


# # Create a railway networks with possible terminal nodes. This returns a complex network, as includes the rail tracks with the highest level of detail.
# aggregation_range = 0.08 # in km
# complex_rail_network = get_rail_network_with_terminals(network_gdf=rail_track_gdf, aggregation_range=aggregation_range)
# output_name = f'study_area{study_area_suffix}'
# with open(root_folder.joinpath(f'networks/complex_rail_network_{output_name}.geojson'), 'wb') as handle:
#     pickle.dump(complex_rail_network, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Creating separate gdf for viasualisation purposes
# non_aggregate_terminals = complex_rail_network.nodes[
#         (complex_rail_network.nodes['terminal_collection'].apply(lambda val: len(val) if isinstance(val,list) else 0) == 1)
#     ]

# aggregate_terminals = complex_rail_network.nodes[
#         (complex_rail_network.nodes['terminal_collection'].apply(lambda val: len(val) if isinstance(val,list) else 0) > 1)
#     ]

# if aggregation_range > 0:
#     demand_edges = complex_rail_network.edges[
#             complex_rail_network.edges['demand_edge'] == 1
#         ]

#     actual_edges = complex_rail_network.edges[
#             complex_rail_network.edges['demand_edge'] == 0
#         ]
# else:
#     actual_edges = complex_rail_network.edges

# # Visualise
# rail_map = actual_edges.explore(tiles="CartoDB positron")
# rail_map = demand_edges.explore(m=rail_map, color='springgreen')

# if len(non_aggregate_terminals)>0:
#     rail_map = non_aggregate_terminals.explore(m=rail_map, color='springgreen', marker_kwds={'radius':3})
# rail_map = aggregate_terminals.explore(m=rail_map, color='darkgreen', marker_kwds={'radius':3})
# rail_map

## merge edges around nodes of degree=2 excluding bridges and tunnels
# merged_rail_network = _merge_edges(network=complex_rail_network, excluded_edge_types=['bridge', 'tunnel']) #Must add network= to pass excluded_edge_types as a keyword argument