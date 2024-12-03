import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hazard_events.hazard_map_utilities import *

# Set the path to the hazard map
fmap=Path(r'C:\Data\Floods\Germany\raw_data\SzenarioHaufig\DERP_RW_L.shp')

fmap_name = str(fmap.stem)

fmap_gdf = validate_hazard_maps(source_path=fmap, hazard_map_name=fmap_name)

fmap_gdf.geometry = fmap_gdf.geometry.simplify(0.000014)

fmap_gdf = simplify_geometries(fmap_gdf, tolerance=0.000014)

output_path = rf'N:\Projects\11209000\11209175\B. Measurements and calculations\Data\hazard_maps_DERP\{fmap_name}_4326_valid_simplified_1m.geojson'

fmap_gdf.to_file(output_path)