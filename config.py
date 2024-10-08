import configparser

config = configparser.ConfigParser()

miraca_colors = {'primary blue':'#4069F6',
                 'accent green':'#64F4C0',
                'white':'#FFFFFF',
                'black':'#171E37',
                'blue_900': '#233778',
                'blue_800': '#2A4396',
                'blue_700': '#314EB3',
                'blue_600': '#385AD1',
                'blue_500': '#4069F6',
                'blue_400': '#6687F8',
                'blue_300': '#94ABFA',
                'blue_200': '#C2CFFC',
                'blue_100': '#E0E7FE',
                'green_900': '#429787',
                'green_800': '#4CB499',
                'green_700': '#56CEA9',
                'green_600': '#5ADBB1',
                'green_500': '#64F4C0',
                'green_400': '#9CF8D7',
                'green_300': '#B5FAE1',
                'green_200': '#CDFCEB',
                'green_100': '#E0FDF2',
                'grey_900': '#373D52',
                'grey_800': '#545866',
                'grey_700': '#676B7A',
                'grey_600': '#7B7F8F',
                'grey_500': '#8F94A3',
                'grey_400': '#A5A9B8',
                'grey_300': '#BCBFCC',
                'grey_200': '#D3D6E0',
                'grey_100': '#EBEDF5',
                'red_danger':'#ED5861',
                'yellow_alert':'#F8CD48',
                'green_success':'#72DA95'}

config['DEFAULT'] = {
    'hazard_type': 'fluvial',
    'infra_type': 'rail',
    'country_code': 'DEU',
    'country_name': 'Germany',
    'hazard_data_subfolders': 'raw_subsample/validated_geometries',
    #'asset_data': 'Exposure/raw_rail_track_study_area_simplify_test.geojson',
    'asset_data': 'Exposure/raw_rail_track_study_area_Rhine_Alpine_DEU.geojson',
    #'asset_data': 'Exposure/raw_rail_track_study_area_KOBLENZ_BONN.geojson',
    #'asset_data': 'Exposure/raw_rail_track_study_area_Rhineland-Palatinate_DEU.geojson',
    'vulnerability_data': 'Vulnerability',
    'miraca_colors': str(miraca_colors)}

with open('config_ci_adapt.ini', 'w') as configfile:
    config.write(configfile)