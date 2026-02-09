##################################
# Reading CSVs
##################################

import seaborn 
import lxml 
import openpyxl
import pandas as pd

filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/addresses.csv"
df = pd.read_csv(filename)

##################################
# Adding Column Names
##################################

df.columns =['First Name', 'Last Name', 'Location ', 'City','State','Area Code']

##################################
# Selecting Column/s
##################################

print(df["First Name"])

df = df[['First Name', 'Last Name', 'Location ', 'City','State','Area Code']]
print(df)

##################################
# Selecting Rows with iloc & loc
##################################

# To select the first row
print(df.loc[0])

# To select the 0th,1st and 2nd row of "First Name" column only
print(df.loc[[0,1,2], "First Name" ])

# To select the 0th,1st and 2nd row of "First Name" column only
print(df.iloc[[0,1,2], 0])


##################################
# Transforming Functions with Pandas
##################################

import pandas as pd
import numpy as np

# Creating a dataframe
df=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
print(df)

# Applying the transform function
df = df.transform(func = lambda x : x + 10)
print(df)

# Finding the square root of each element
result = df.transform(func = ['sqrt'])
print(result)

##################################
# JSON file Format
##################################

import json

person = {
    'first_name' : 'Mark',
    'last_name' : 'abc',
    'age' : 27,
    'address': {
        "streetAddress": "21 2nd Street",
        "city": "New York",
        "state": "NY",
        "postalCode": "10021-3100"
    }
}

##################################
# Writing to JSON
##################################

#json.dump() used for writing as well
with open('person.json', 'w') as f:  # writing JSON object
    json.dump(person, f)

    # Serializing json  
json_object = json.dumps(person, indent = 4) 
  
# Writing to sample.json 
with open("sample.json", "w") as outfile: 
    outfile.write(json_object) 

print(json_object)

##################################
# Reading JSON
##################################

# Opening JSON file 
with open('sample.json', 'r') as openfile: 
  
    # Reading from json file 
    json_object = json.load(openfile) 
  
print(json_object) 
print(type(json_object))

filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/file_example_XLSX_10.xlsx"

df2 = pd.read_excel(filename)

print(df2)

##################################
# XML File Format
##################################

import xml.etree.ElementTree as ET

# create the file structure
employee = ET.Element('employee')
details = ET.SubElement(employee, 'details')
first = ET.SubElement(details, 'firstname')
second = ET.SubElement(details, 'lastname')
third = ET.SubElement(details, 'age')
first.text = 'Shiv'
second.text = 'Mishra'
third.text = '23'

##################################
# Writing to XML Format
##################################

# create a new XML file with the results
mydata1 = ET.ElementTree(employee)
# myfile = open("items2.xml", "wb")
# myfile.write(mydata)
with open("new_sample.xml", "wb") as files:
    mydata1.write(files)


##################################
# Reading XML Format
##################################

import requests
import xml.etree.ElementTree as ET

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/Sample-employee-XML-file.xml"

response = requests.get(url)

with open("Sample-employee-XML-file.xml", "wb") as f:
    f.write(response.content)

# Parse the XML file
tree = ET.parse("Sample-employee-XML-file.xml")

# Get the root of the XML tree
root = tree.getroot()

# Define the columns for the DataFrame
columns = ["firstname", "lastname", "title", "division", "building", "room"]

# Initialize an empty DataFrame
datatframe = pd.DataFrame(columns=columns)

# Iterate through each node in the XML root
for node in root:
    # Extract text from each element
    firstname = node.find("firstname").text
    lastname = node.find("lastname").text
    title = node.find("title").text
    division = node.find("division").text
    building = node.find("building").text
    room = node.find("room").text
    
    # Create a DataFrame for the current row
    row_df = pd.DataFrame([[firstname, lastname, title, division, building, room]], columns=columns)
    
    # Concatenate with the existing DataFrame
    datatframe = pd.concat([datatframe, row_df], ignore_index=True)

    print(datatframe)

##################################
# Reading XML Format with Pandas
##################################

# Herein xpath we mention the set of xml nodes to be considered for migrating  to the dataframe which in this case is details node under employees.
df=pd.read_xml("Sample-employee-XML-file.xml", xpath="/employees/details") 

##################################
# Save Data
##################################

datatframe.to_csv("employee.csv", index=False)

##################################
# Binary File Format
##################################

from PIL import Image
import urllib.request

image_url = "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg"
local_filename = "dog.jpg"

urllib.request.urlretrieve(image_url, local_filename)

img = Image.open(local_filename)
img.show()


##################################
# Data Analysis
##################################

filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-PY0101EN-SkillsNetwork/labs/Module%205/data/diabetes.csv"
df3 = pd.read_csv(filename)

# show the first 5 rows using dataframe.head() method
print("The first 5 rows of the dataframe") 
print(df3.head(5))

##################################
# Statistical Overview of Dataset
##################################
print(df3.info)
print(df3.describe())

missing_data = df3.isnull()
print(missing_data.head(5))

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    

##################################
# Correct Data Format
##################################
print(df3.dtypes)

##################################
# Visualisation
##################################
import matplotlib.pyplot as plt
import seaborn as sns

labels= 'Not Diabetic','Diabetic'
plt.pie(df3['Outcome'].value_counts(),labels=labels,autopct='%0.02f%%')
plt.legend()
plt.show()