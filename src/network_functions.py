import snkit
import networkx as nx
from networkx import MultiGraph
from pyproj import CRS
from typing import Union, List, Optional

# These network functions are based on Tom Russell's snkit library and code from Asgarpour, SA.
def _network_to_nx(
    net: snkit.network.Network,
    node_id_column_name="id",
    edge_from_id_column="from_id",
    edge_to_id_column="to_id",
    default_crs: Optional[CRS] = CRS.from_epsg(4326),
) -> MultiGraph:
    g = nx.MultiGraph()

    # Add nodes to the graph
    for index, row in net.nodes.iterrows():
        node_id = row[node_id_column_name]
        attributes = {k: v for k, v in row.items()}
        g.add_node(node_id, **attributes)

    # Add edges to the graph
    for index, row in net.edges.iterrows():
        u = row[edge_from_id_column]
        v = row[edge_to_id_column]
        attributes = {k: v for k, v in row.items()}
        g.add_edge(u, v, **attributes)

    # Add CRS information to the graph
    if "crs" not in g.graph:
        g.graph["crs"] = default_crs

    return g