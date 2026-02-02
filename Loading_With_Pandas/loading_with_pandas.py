#pip install xlrd openpyxl
#/usr/local/bin/python3 -m pip install openpyxl
import pandas as pd

####################################
# Read a CSV using Pandas
####################################

csv_path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/LXjSAttmoxJfEG6il1Bqfw/Product-sales.csv'
df = pd.read_csv(csv_path)

print(df.head())

####################################
# Read an Excel Sheet Using Pandas
####################################

xlsx_path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/n9LOuKI9SlUa1b5zkaCMeg/Product-sales.xlsx'
df = pd.read_excel(xlsx_path)

print(df.head())

####################################
# Accessing Columns
####################################
# Get the column as a series

x = df['Product']
print(x)

# Get the column as a dataframe
x = df[['Quantity']]
type(x)

# Access to multiple columns
y = df[['Product','Category', 'Quantity']]
print(y)

####################################
# iloc Method
####################################

# Access the value on the first row and the first column
print(df.iloc[0, 0])

# Access the value on the second row and the first column
print(df.iloc[1,0])

# Access the value on the first row and the third column
print(df.iloc[0,2])

# Access the value on the second row and the third column
print(df.iloc[1,2])

####################################
# loc Method
####################################

# Access the column using the name
print(df.loc[0, 'Product'])

# Access the column using the name
print(df.loc[1, 'Product'])

# Access the column using the name
print(df.loc[1, 'CustomerCity'])

# Access the column using the name
print(df.loc[1, 'Total'])


####################################
# Slicing
####################################

# Slicing the dataframe
print(df.iloc[0:2, 0:3])

# Slicing the dataframe using name
print(df.loc[0:2, 'OrderID':'Category'])

