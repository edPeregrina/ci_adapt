import geopandas as gpd
from shapely import Point

class GSNetwork:
    def __init__(self,gdf_sources,gdf_sinks,buffer=0) -> None:
        self.gdf_sources=gdf_sources
        self.gdf_sinks=gdf_sinks
        self.bbox=create_bounding_box(gdf_sinks,gdf_sources,buffer)

    def retrieve_demand_sinks(gdf_sources):



    

def create_bounding_box(gdf1,gdf2, buffer=0): #TODO MOVE TO UTILITIES
        min_x = min(gdf1.total_bounds[0], gdf2.total_bounds[0]) - buffer
        min_y = min(gdf1.total_bounds[1], gdf2.total_bounds[1]) - buffer
        max_x = max(gdf1.total_bounds[2], gdf2.total_bounds[2]) + buffer
        max_y = max(gdf1.total_bounds[3], gdf2.total_bounds[3]) + buffer
        return (min_x, min_y, max_x, max_y)


# Example GeoDataFrame s
data1 = {'geometry': [Point(10, 30), Point(20, 25), Point(15, 35), Point(25, 41)]}
data2 = {'geometry': [Point(9, 30), Point(20, 25), Point(15, 35), Point(25, 40)]}
gdf_sources = gpd.GeoDataFrame(data1, geometry='geometry')
gdf_sinks = gpd.GeoDataFrame(data2, geometry='geometry')

# Example usage
gs_nw_bbox = GSNetwork(gdf_sources,gdf_sinks,buffer=5)
print(gs_nw_bbox.bbox)



    def retrieve_demand_sinks(gdf_sinks):
        for e in gdf_sinks try:
            if e is a list: find geometries and make gdf_sinks
            elif e is a gdf: pass

            if within bbox: pass
            else: trigger warning and specific search

        
            if row.geometry in [Point, Line]: repeat search with 'area' keyword
                
            elif row.geometry in [Polygon, MultiPoligon]: 
                return 'lol'
        

    
        
        
                




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