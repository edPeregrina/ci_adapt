####
##WORK IN PROGRESS, USE WITH CAUTION
###
import pickle
import numpy as np
# from shapely.validation import make_valid # only needed to make invalid geometries valid
from ci_adapt_utilities import *
from ci_adapt_classes import *

# configure handler with ini file. ini file can be created with config.py
H=Handler(config_file='config_ci_adapt.ini')
# read vulnerability and maximum damage data
H.read_vul_maxdam()

# read hazard data
H.read_hazard_data()

# read exposure data
H.read_asset_data()

# calculate direct damage by asset
# currently data with 3 baseline return periods: H=RP10, M=RP100, L=RP200
# TODO: for DEXX_RP, return period should be input or taken from dictionary/file names
# TODO: verify cost of damage and adaptation regarding double rail or single rail, divide /2 if double rail? 
H.run_direct_damage()
pd.DataFrame.from_dict(H.collect_output).to_csv(H.interim_data_path / 'sample_collected_run.csv')

# calculate EAD by year (timestep)
H.ead=H.calculate_ead()
ead_by_ts_plot(H.ead)


# TESTING ADAPTATION
# create a variable changed_assets
changed_asset_list = [105426, 110740, 118116] # testing with a few assets

# open pickled hazard-asset overlay and hazard intensity data
with open(H.interim_data_path / 'overlay_assets_flood_DERP_RW_L_4326_2080411370.pkl', 'rb') as f:
    overlay_assets = pickle.load(f)
with open(H.interim_data_path / 'hazard_numpified_flood_DERP_RW_L_4326_2080411370.pkl', 'rb') as f:
    hazard_numpified_list = pickle.load(f)

# optionally to calculate the maximum intensity for each hazard point, this can be used, else a float can be used
max_intensity = []
for asset_id in changed_asset_list:
    max_intensity.append(retrieve_max_intensity_by_asset(asset_id, overlay_assets, hazard_numpified_list))

# extract the assets that will be adapted
changed_assets = H.assets.assets.loc[H.assets.assets.index.isin(changed_asset_list)].copy() # testing with a few assets
# add new columns fragility_mod and haz_mod
changed_assets['fragility_mod'] = 1 #[0.3, 0.5, 0.8] #fraction (1 = no reduction, 0 = invulnerable asset) DUMMY DATA FOR TESTING
changed_assets['haz_mod'] = [np.max(x)+1 for x in max_intensity] #meters (0 = no reduction in hazard intensity, 0.5 = 0.5 meter reduction in hazard intensity) DUMMY DATA FOR TESTING consider raising railway 0.5 meters

# TODO: automate infrastructure curve deduction from dictionary keys
hazard_intensity = H.infra_curves['F8.1'].index.values
fragility_values = (np.nan_to_num(H.infra_curves['F8.1'].values,nan=(np.nanmax(H.infra_curves['F8.1'].values)))).flatten()
maxdams_filt=H.max_damage_tables[H.max_damage_tables['ID number']=='F8.1']['Amount']

adaptation_run = run_damage_reduction_by_asset(H.assets.geom_dict, overlay_assets, hazard_numpified_list, changed_assets, hazard_intensity, fragility_values, maxdams_filt)

#reporting
for asset_id, baseline_damages in adaptation_run[0].items():
    print(f'\nADAPTATION results for asset {asset_id}:')
    print(f'Baseline damages for asset {asset_id}: {baseline_damages}')
    print(f'Adapted damages for asset {asset_id}: {adaptation_run[1][asset_id]}')
    delta = tuple(adaptation_run[1][asset_id][i] - baseline_damages[i] for i in range(len(baseline_damages)))
    percent_change = tuple((100 * (delta[i] / baseline_damages[i])) for i in range(len(baseline_damages)))

    print(f'Change (Adapted-Baseline): {delta}, {percent_change}% change, at a cost of {adaptation_run[2][asset_id]}')

#TODO Check with economist: ammortization of adaptation cost over years of adaptation scenario
    #NEXT: adaptation_run returns (collect_inb_bl, collect_inb_adapt, adaptation_cost). These can be used to calculate the EAD for the adapted scenario (and damage reduction), and compare with the adaptation cost, which must be ammortized over the years of the adaptation scenario.


