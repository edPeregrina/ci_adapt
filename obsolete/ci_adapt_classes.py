import configparser
from pathlib import Path
import pathlib
from direct_damages import damagescanner_rail_track as ds
import pandas as pd
import geopandas as gpd
import datetime
from ci_adapt_utilities import *
import pickle



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

    def calculate_ead(self):
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
        return_period_dict = {
            '_H_': 10,
            '_M_': 100,
            '_L_': 200
        }
        increase_factor = { #1.2 to 9.0, Q3 Kees
            '_H_': (1.2+9.0)/2,
            '_M_': (1.2+9.0)/2,
            '_L_': (1.2+9.0)/2 #average of the range (1.2 to 9.0) based on change since 1900
        }

        num_years = 100

        return_periods = calculate_dynamic_return_periods(return_period_dict, num_years, increase_factor)

        # define dictionary to relate water depth classes to water depths, specific per region, in this case Rheinland Palatinate is used
        return_period_dict = {}
        return_period_dict['DERP'] = return_periods #{ #TODO: make generic
        #     '_H_': [10, 5],
        #     '_M_': [100, 50],
        #     '_L_': [200, 100]
        # }

        # add the return period column to aggregated_df
        aggregated_df['Return Period'] = [return_period_dict['DERP'][index] for index in aggregated_df.index]
        print(aggregated_df)

        # sort the DataFrame by return period
        aggregated_df = aggregated_df.sort_values('Return Period', ascending=True)

        # Calculate the probability of each return period
        aggregated_df['Probability'] = [[1 / x for x in i] for i in aggregated_df['Return Period']]
        probabilities = aggregated_df['Probability']
        dmgs = []
        for ts in range(len(probabilities.iloc[0])):    
            dmgs_l = []
            dmgs_u = []

            for rp in range(len(probabilities)-1):
                d_rp= probabilities.iloc[rp][ts] - probabilities.iloc[rp + 1][ts]
                mean_damage_l = 0.5 * (aggregated_df['Total Damage Lower Bound'].iloc[rp] + aggregated_df['Total Damage Lower Bound'].iloc[rp + 1])
                mean_damage_u = 0.5 * (aggregated_df['Total Damage Upper Bound'].iloc[rp] + aggregated_df['Total Damage Upper Bound'].iloc[rp + 1])
                dmgs_l.append(d_rp * mean_damage_l)
                dmgs_u.append(d_rp * mean_damage_u)
            
            # adding the portion of damages corresponding to p=0 to p=1/highest return period
            # This calculation considers the damage for return periods higher than the highest return period the same as the highest return period
            d0_rp = probabilities.iloc[-1][ts]
            mean_damage_l0 = max(aggregated_df['Total Damage Lower Bound'])
            mean_damage_u0 = max(aggregated_df['Total Damage Upper Bound'])
            dmgs_l.append(d0_rp * mean_damage_l0)
            dmgs_u.append(d0_rp * mean_damage_u0)

            # This calculation considers that no assets are damaged at a return period of 4 years
            d_end_rp = (1/4)-probabilities.iloc[0][ts]
            mean_damage_l_end = 0.5 * min(aggregated_df['Total Damage Lower Bound'])
            mean_damage_u_end = 0.5 * min(aggregated_df['Total Damage Upper Bound'])
            dmgs_l.append(d_end_rp * mean_damage_l_end)
            dmgs_u.append(d_end_rp * mean_damage_u_end)

            dmgs.append((sum(dmgs_l), sum(dmgs_u)))
        
        ead_by_ts = pd.DataFrame(dmgs, columns=['Total Damage Lower Bound', 'Total Damage Upper Bound'])

        print(f'Baseline expected annual damages: {dmgs[0]}')
        print(f'Expected annual damages by year {num_years}: {dmgs[-1]}')
        return ead_by_ts


class Assets:
    def __init__(self, file_path):
        self.assets = gpd.read_file(file_path)
        self.assets = gpd.GeoDataFrame(self.assets).set_crs(4326).to_crs(3857)
        self.assets = self.assets.loc[self.assets.geometry.geom_type == 'LineString']
        self.assets = self.assets.rename(columns={'railway' : 'asset'})

        #TODO: add reset index and test 
        # Uncomment the following lines if you want to drop passenger lines and light rails
        #self.assets = self.assets.loc[~(self.assets['railway:traffic_mode'] == 'passenger')]
        #self.assets = self.assets.loc[~(self.assets['asset'] == 'light_rail')]

        # Uncomment the following lines if you want to drop bridges and tunnels
        #self.assets = self.assets.loc[~(self.assets['bridge'].isin(['yes']))]
        #self.assets = self.assets.loc[~(self.assets['tunnel'].isin(['yes']))]

        self.buffered_assets = ds.buffer_assets(self.assets)
        self.geom_dict = self.assets['geometry'].to_dict()
        self.type_dict = self.assets['asset'].to_dict()
