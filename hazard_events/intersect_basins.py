import os
from qgis.core import QgsApplication

# Set the path to the QGIS installation
qgis_path = r'C:\Users\peregrin\AppData\Local\anaconda3\envs\qgis_env\Library\bin\qgis.exe'

# Set the environment variables for QGIS
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(qgis_path, 'apps', 'Qt5', 'plugins', 'platforms')
os.environ['PATH'] += os.pathsep + os.path.join(qgis_path, 'apps', 'qgis', 'bin')
os.environ['PYTHONPATH'] += os.pathsep + os.path.join(qgis_path, 'apps', 'qgis', 'python')
os.environ['LD_LIBRARY_PATH'] = os.path.join(qgis_path, 'apps', 'qgis', 'lib')

# Initialize QGIS application
QgsApplication.setPrefixPath(qgis_path, True)
qgs = QgsApplication([], False)
qgs.initQgis()

# Now you can use QGIS processing tools
from qgis.analysis import QgsNativeAlgorithms
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

import processing
# intersect basin polygons with flood map
processing.run("native:intersection", {'INPUT':'C:/Data/Floods/Germany/interim_data/DERP_RW_L_4326_valid_lmu.geojson','OVERLAY':'C:/Data/Floods/basins/hybas_eu_lev01-12_v1c/hybas_eu_lev08_v1c_valid.shp','INPUT_FIELDS':['flood_area','depth_class','w_depth_l', 'w_depth_m', 'w_depth_u'],'OVERLAY_FIELDS':['HYBAS_ID','NEXT_DOWN','NEXT_SINK','MAIN_BAS','DIST_SINK','DIST_MAIN','SUB_AREA','UP_AREA','PFAF_ID','ORDER'],'OVERLAY_FIELDS_PREFIX':'','OUTPUT':'C:/Data/Floods/Germany/basin_intersections/DERP_RW_L_4326_hybas_intersections.geojson','GRID_SIZE':None})

# exit QGIS application
qgs.exitQgis()