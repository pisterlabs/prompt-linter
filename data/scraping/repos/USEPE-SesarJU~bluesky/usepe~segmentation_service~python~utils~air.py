import os
# from python.utils import openair
from usepe.segmentation_service.python.utils import openair
import geopandas as gpd
import osmnx as ox
import pandas as pd


_ASP_FILE = os.path.join( 
    os.path.abspath( os.path.dirname( __file__ ) ), "../../data/airspace/de_asp.txt"
 )


def add_restrictions( airspaces, rules ):
    airspaces["z_min"] = False
    airspaces["z_max"] = False
    airspaces["speed_min"] = False
    airspaces["speed_max"] = False
    for name, data in rules["classes"].items():
        airspaces.loc[airspaces["class"] == name, "z_min"] = min( data["altitude"] )
        airspaces.loc[airspaces["class"] == name, "z_max"] = max( data["altitude"] )
        airspaces.loc[airspaces["class"] == name, "speed_min"] = min( data["velocity"] )
        airspaces.loc[airspaces["class"] == name, "speed_max"] = max( data["velocity"] )
    return airspaces


def get( region, rules ):
    airspaces = openair.to_gdf( _ASP_FILE )
    airspaces = gpd.sjoin( 
        airspaces,
        gpd.GeoDataFrame( {"label": ["in_region"], "geometry": [region]} ),
        "inner",
        "intersects",
    )
    airspaces = airspaces[airspaces["z_min"] < max( rules["vll"] )]
    airspaces = airspaces.rename( columns={"class": "type"} )
    airspaces["class"] = "grey"
    airspaces["element_type"] = "airspace"
    airspaces["buffer"] = 0
    airspaces = airspaces.set_index( "element_type", append=True )
    airspaces = airspaces.rename_axis( index=["id", "element"] ).reorder_levels( [1, 0] )

    airspaces = add_restrictions( airspaces, rules )

    return airspaces[
        ["class", "type", "name", "geometry", "buffer", "z_min", "z_max", "speed_min", "speed_max"]
    ]
