import configparser

config = configparser.ConfigParser()

config['DEFAULT'] = {
    'hazard_type': 'fluvial',
    'infra_type': 'rail',
    'country_code': 'DEU',
    'country_name': 'Germany',
    'hazard_data_subfolders': 'raw_subsample/validated_geometries',
    'asset_data': 'Exposure/raw_rail_track_study_area_KOBLENZ_BONN.geojson',#Exposure/raw_rail_track_study_area_Rhine_Alpine_DEU.geojson', 
    'vulnerability_data': 'Vulnerability'}

with open('config_ci_adapt.ini', 'w') as configfile:
    config.write(configfile)