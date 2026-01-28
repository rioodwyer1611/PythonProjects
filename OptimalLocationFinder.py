#!pip install geopy
#!pip install requests
#!pip install folium
#!pip install ortools


###################################
# Import Statements.
###################################
import sys
import docplex.mp
import geopy.distance
import ortools
from geopy.distance import great_circle
import requests
import json
import folium
import webbrowser
import os
from ortools.sat.python import cp_model


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
# Save for use in a web browser.
###################################

# Define the file path and save the map as an HTML file
filepath = 'folium_map.html'
map_osm.save(filepath)



# This code was written with the IMB Data Science Professional Certificate in mind, which is able to use CPLEX.
# This version/storage of the project cannot.
"""

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

"""
# Below is an OR TOOLS rewrite of the CPLEX solition.

###################################
# Precompute Distances
###################################

dist = {}
for i, c in enumerate(libraries):
    for j, b in enumerate(libraries):
        dist[i, j] = int(1000 * get_distance(c, b))

###################################
# OR-Tools Model
###################################

model = cp_model.CpModel()

N = len(libraries)

# Decision variables
open_shop = [model.NewBoolVar(f"open_{i}") for i in range(N)]
assign = {}
for i in range(N):
    for j in range(N):
        assign[i, j] = model.NewBoolVar(f"assign_{i}_{j}")

###################################
# Constraints
###################################

# Each library assigned to exactly one coffee shop
for j in range(N):
    model.Add(sum(assign[i, j] for i in range(N)) == 1)

# Assignment only if coffee shop is open
for i in range(N):
    for j in range(N):
        model.Add(assign[i, j] <= open_shop[i])

# Exactly nb_shops coffee shops open
model.Add(sum(open_shop[i] for i in range(N)) == nb_shops)

###################################
# Objective: minimize total distance
###################################

model.Minimize(
    sum(assign[i, j] * dist[i, j] for i in range(N) for j in range(N))
)

###################################
# Solve
###################################

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30
solver.parameters.num_search_workers = 8

status = solver.Solve(model)

assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

###################################
# Extract Solution
###################################

open_shops = [libraries[i] for i in range(N) if solver.Value(open_shop[i]) == 1]
edges = []

for j in range(N):
    for i in range(N):
        if solver.Value(assign[i, j]) == 1:
            edges.append((libraries[i], libraries[j]))

print(f"Total distance = {solver.ObjectiveValue() / 1000:.2f} miles")
print(f"# coffee shops = {len(open_shops)}")

for s in open_shops:
    print(f"New coffee shop: {s}")

###################################
# Visualization
###################################

map_osm = folium.Map(location=[41.878, -87.629], zoom_start=11)

# Coffee shops in red
for s in open_shops:
    folium.Marker([s.y, s.x],
                  icon=folium.Icon(color="red", icon="info-sign")).add_to(map_osm)

# Libraries in blue
for b in libraries:
    if b not in open_shops:
        folium.Marker([b.y, b.x]).add_to(map_osm)

# Assignment edges
for c, b in edges:
    folium.PolyLine([[c.y, c.x], [b.y, b.x]],
                    color="red",
                    weight=3).add_to(map_osm)

filepath = "folium_map.html"
map_osm.save(filepath)
webbrowser.open("file://" + os.path.realpath(filepath))