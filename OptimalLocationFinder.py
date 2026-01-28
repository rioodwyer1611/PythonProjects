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
from docplex.mp.model import Model


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


###################################
# Prescriptive Model Setup.
###################################

env = Environment()
env.print_information()

mdl = Model("coffee shops")

BIGNUM = 999999999

# Ensure unique points
libraries = set(libraries)
# For simplicity, let's consider that coffee shops candidate locations are the same as libraries locations.
# That is: any library location can also be selected as a coffee shop.
coffeeshop_locations = libraries

# Decision vars
# Binary vars indicating which coffee shop locations will be actually selected
coffeeshop_vars = mdl.binary_var_dict(coffeeshop_locations, name="is_coffeeshop")
#
# Binary vars representing the "assigned" libraries for each coffee shop
link_vars = mdl.binary_var_matrix(coffeeshop_locations, libraries, "link")

###################################
# Express Business Constraints.
###################################

# First constraint: if the distance is suspect, it needs to be excluded from the problem.
for c_loc in coffeeshop_locations:
    for b in libraries:
        if get_distance(c_loc, b) >= BIGNUM:
            mdl.add_constraint(link_vars[c_loc, b] == 0, "ct_forbid_{0!s}_{1!s}".format(c_loc, b))

# Second constraint: each library must be linked to a coffee shop that is open.
mdl.add_constraints(link_vars[c_loc, b] <= coffeeshop_vars[c_loc]
                   for b in libraries
                   for c_loc in coffeeshop_locations)
mdl.print_information()

# Third constraint: each library is linked to exactly one coffee shop.
mdl.add_constraints(mdl.sum(link_vars[c_loc, b] for c_loc in coffeeshop_locations) == 1
                   for b in libraries)
mdl.print_information()

# Fourth constraint: there is a fixed number of coffee shops to open.
# Total nb of open coffee shops
mdl.add_constraint(mdl.sum(coffeeshop_vars[c_loc] for c_loc in coffeeshop_locations) == nb_shops)

# Print model information
mdl.print_information()

###################################
# Express the Objective.
###################################

# Minimize total distance from points to hubs
total_distance = mdl.sum(link_vars[c_loc, b] * get_distance(c_loc, b) for c_loc in coffeeshop_locations for b in libraries)
mdl.minimize(total_distance)

###################################
# Solve the Model.
###################################

print("# coffee shops locations = %d" % len(coffeeshop_locations))
print("# coffee shops           = %d" % nb_shops)

assert mdl.solve(), "!!! Solve of the model fails"

###################################
# Investigate Solution.
###################################

total_distance = mdl.objective_value
open_coffeeshops = [c_loc for c_loc in coffeeshop_locations if coffeeshop_vars[c_loc].solution_value == 1]
not_coffeeshops = [c_loc for c_loc in coffeeshop_locations if c_loc not in open_coffeeshops]
edges = [(c_loc, b) for b in libraries for c_loc in coffeeshop_locations if int(link_vars[c_loc, b]) == 1]

print("Total distance = %g" % total_distance)
print("# coffee shops  = {0}".format(len(open_coffeeshops)))
for c in open_coffeeshops:
    print("new coffee shop: {0!s}".format(c))

###################################
# Display Solution on Map.
###################################

map_osm = folium.Map(location=[41.878, -87.629], zoom_start=11)
for coffeeshop in open_coffeeshops:
    lt = coffeeshop.y
    lg = coffeeshop.x
    folium.Marker([lt, lg], icon=folium.Icon(color='red',icon='info-sign')).add_to(map_osm)
    
for b in libraries:
    if b not in open_coffeeshops:
        lt = b.y
        lg = b.x
        folium.Marker([lt, lg]).add_to(map_osm)
    

for (c, b) in edges:
    coordinates = [[c.y, c.x], [b.y, b.x]]
    map_osm.add_child(folium.PolyLine(coordinates, color='#FF0000', weight=5))

###################################
# Open in a web browser.
###################################

# Define the file path and save the map as an HTML file
filepath = 'folium_map.html'
map_osm.save(filepath)

# Open the HTML file in the default web browser
# The 'file://' prefix is needed for cross-platform compatibility
webbrowser.open('file://' + os.path.realpath(filepath))
