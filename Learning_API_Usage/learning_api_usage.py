# /usr/local/bin/python3 -m pip install randomuser
# pip install pandas

#######################################
# RandomUsers Package
#######################################

from randomuser import RandomUser
import pandas as pd

r = RandomUser()

#######################################
# Generate a list of 10 random users
#######################################

some_list = r.generate_users(10)
print(some_list)

name = r.get_full_name()

#######################################
# Generate user data
#######################################

for user in some_list:
    print (user.get_full_name()," ",user.get_email())

for user in some_list:
     print (user.get_picture())

#######################################
# Generate a table of the user data
#######################################

def get_users():
    users =[]
     
    for user in RandomUser.generate_users(10):
        users.append({"Name":user.get_full_name(),"Gender":user.get_gender(),"City":user.get_city(),"State":user.get_state(),"Email":user.get_email(), "DOB":user.get_dob(),"Picture":user.get_picture()})
      
    return pd.DataFrame(users)     

df1 = pd.DataFrame(get_users())  
print(get_users())


#######################################
# Fruityvice API
#######################################

import requests
import json

data = requests.get("https://web.archive.org/web/20240929211114/https://fruityvice.com/api/fruit/all")

results = json.loads(data.text)

pd.DataFrame(results)
df2 = pd.json_normalize(results)

#######################################
# Find Family and Genus of a Cherry
#######################################

cherry = df2.loc[df2["name"] == 'Cherry']
print((cherry.iloc[0]['family']) , (cherry.iloc[0]['genus']))

#######################################
# Find Calories of Banana
#######################################

banana = df2.loc[df2["name"] == 'Banana']
print(banana.iloc[0]['nutritions.calories'])

#######################################
# Official Joke API
#######################################

url = "https://official-joke-api.appspot.com/jokes/ten"
data2 = requests.get(url)

results2 = json.loads(data2.text)

df3 = pd.DataFrame(results2)
df3.drop(columns=["type","id"], inplace=True)
print(df3)