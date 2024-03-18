import pickle
from from_elco import damagescanner_rail_track as ds

def pickle_overlay_hazard(overlay_assets, hazard_numpified, damage_curve='8.1'):
    with open('overlay_assets.pkl', 'wb') as f:
        pickle.dump(overlay_assets, f)
    with open('numpified_hazard.pkl', 'wb') as f:
        pickle.dump(hazard_numpified, f)
    with open('damage_curve.pkl', 'wb') as f:
        pickle.dump(damage_curve, f)


## Check if the pickle files exist, load pickle and move to next file in the list if they do
# if Path(hazard_numpified_path).is_file() and Path(overlay_path).is_file():
#     print('Flood maps found in pickle files will be loaded')
#     # If they do, load the data from the pickle files
#     with open(hazard_numpified_path, 'rb') as f:
#         hazard_numpified_list = pickle.load(f)
#     with open(overlay_path, 'rb') as f:
#         overlay_assets = pickle.load(f)                
#     continue

# else: pass





# import geopandas as gpd
# from shapely import Point

# class GSNetwork:
#     def __init__(self,gdf_sources,gdf_sinks,buffer=0) -> None:
#         self.gdf_sources=gdf_sources
#         self.gdf_sinks=gdf_sinks
#         self.bbox=create_bounding_box(gdf_sinks,gdf_sources,buffer)

#     def retrieve_demand_sinks(gdf_sources):

# def create_bounding_box(gdf1,gdf2, buffer=0): #TODO MOVE TO UTILITIES
#         min_x = min(gdf1.total_bounds[0], gdf2.total_bounds[0]) - buffer
#         min_y = min(gdf1.total_bounds[1], gdf2.total_bounds[1]) - buffer
#         max_x = max(gdf1.total_bounds[2], gdf2.total_bounds[2]) + buffer
#         max_y = max(gdf1.total_bounds[3], gdf2.total_bounds[3]) + buffer
#         return (min_x, min_y, max_x, max_y)


# # Example GeoDataFrame s
# data1 = {'geometry': [Point(10, 30), Point(20, 25), Point(15, 35), Point(25, 41)]}
# data2 = {'geometry': [Point(9, 30), Point(20, 25), Point(15, 35), Point(25, 40)]}
# gdf_sources = gpd.GeoDataFrame(data1, geometry='geometry')
# gdf_sinks = gpd.GeoDataFrame(data2, geometry='geometry')

# # Example usage
# gs_nw_bbox = GSNetwork(gdf_sources,gdf_sinks,buffer=5)
# print(gs_nw_bbox.bbox)




    
        
        



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








                




#         gdf_sinks_demand=gdf_sinks #+input
#         return gdf_sinks_demand
    
#     def retrieve_supply_sources(gdf_sources):
#         gdf_sources_supply=gdf_sources #+input
#         return gdf_sources_supply

# """
# These are utility functions that can be migrated to external utility file
# """  
#     # def get_demand(self,date=None):

# gdf_sinks='a'
# gdf_sources='b'   
# gs_network=GSNetwork(gdf_sinks,gdf_sources)
# # class ClassName:
# #     <statement-1>
# #     .
# #     .
# #     .
# #     <statement-N>

# # class Complex:
# #     def __init__(self, realpart, imagpart):
# #         self.r = realpart
# #         self.i = imagpart

# # x = Complex(3.0, -4.5)
# # x.r, x.i
# # (3.0, -4.5)

# # def scope_test():
# #     def do_local():
# #         spam = "local spam"

# #     def do_nonlocal():
# #         nonlocal spam
# #         spam = "nonlocal spam"

# #     def do_global():
# #         global spam
# #         spam = "global spam"

# #     spam = "test spam"
# #     do_local()
# #     print("After local assignment:", spam)
# #     do_nonlocal()
# #     print("After nonlocal assignment:", spam)
# #     do_global()
# #     print("After global assignment:", spam)

# # scope_test()
# # print("In global scope:", spam)