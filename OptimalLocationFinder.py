#!pip install geopy
#pip install docplex


###################################
# Import Statements.
###################################
import sys
import docplex.mp
import geopy.distance
from geopy.distance import great_circle
import requests
import json
import folium
import webbrowser
import os
from docplex.mp.environment import Environment


###################################
# Class Definitions.
###################################

class XPoint(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return "P(%g_%g)" % (self.x, self.y)

class NamedPoint(XPoint):
    def __init__(self, name, x, y):
        XPoint.__init__(self, x, y)
        self.name = name
    def __str__(self):
        return self.name
    
###################################
# Function Definitions.
###################################

def get_distance(p1, p2):
    return great_circle((p1.y, p1.x), (p2.y, p2.x)).miles


def build_libraries_from_url(url, name_pos, lat_long_pos):
    import requests
    import json

    r = requests.get(url)
    myjson = json.loads(r.text, parse_constant='utf-8')
    myjson = myjson['data']

    libraries = []
    k = 1
    for location in myjson:
        uname = location[name_pos]
        try:
            latitude = float(location[lat_long_pos][1])
            longitude = float(location[lat_long_pos][2])
        except TypeError:
            latitude = longitude = None
        try:
            name = str(uname)
        except:
            name = "???"
        name = "P_%s_%d" % (name, k)
        if latitude and longitude:
            cp = NamedPoint(name, longitude, latitude)
            libraries.append(cp)
            k += 1
    return libraries

###################################
# Build Library Locations 
# from Provided URL.
###################################
url = 'https://data.cityofchicago.org/api/views/x8fc-8rcq/rows.json?accessType=DOWNLOAD'

libraries = build_libraries_from_url(url, 10, 17)

print("There are %d public libraries in Chicago" % (len(libraries)))


nb_shops = 5
print("We would like to open %d coffee shops" % nb_shops)

###################################
# Create Folium Map.
###################################

map_osm = folium.Map(location=[41.878, -87.629], zoom_start=11)
for library in libraries:
    lt = library.y
    lg = library.x
    folium.Marker([lt, lg]).add_to(map_osm)


###################################
# Open in a web browser.
###################################

# Define the file path and save the map as an HTML file
filepath = 'folium_map.html'
map_osm.save(filepath)

# Open the HTML file in the default web browser
# The 'file://' prefix is needed for cross-platform compatibility
webbrowser.open('file://' + os.path.realpath(filepath))