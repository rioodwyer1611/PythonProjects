# mamba install pandas==1.3.3-y
# mamba install numpy=1.21.2-y
# mamba install sklearn=0.20.1-y
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression

file_path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
df = pd.read_csv(file_path)
print(df.head())

#####################################################
# Linear Regression Regression
#####################################################
lm = LinearRegression()
print(lm)

###############################
# Highway-mpg to predict
# Car Price
###############################

# Predictor Variable
X = df[['highway-mpg']]
# Target Variable
Y = df[['price']]

# Fit Linear Model
lm.fit(X,Y)

# Output Prediction
Yhat = lm.predict(X)
Yhat[0:5]

# Intercept Value
print(lm.intercept_)

# Slope Value
print(lm.coef_)

# Should achieve model with equation similar to:
# Yhat = a + bX

###############################
# Engine-size to predict
# Car Price
###############################

# Create Linear Regression Model
lm1 = LinearRegression()

# Predictor Variable
X1 = df[['engine-size']]
# Target Variable
Y1 = df[['price']]

# Fit Linear Model
lm1.fit(X1,Y1)

# Intercept Value
print(lm1.intercept_)

# Slope Value
print(lm1.coef_)

# Equation of predicted line using X and Y  
Yhat=-7963.34 + 166.86*X
Price=-7963.34 + 166.86*df['engine-size']

#####################################################
# Multiple Linear Regression
#####################################################

###############################
# Horsepower, Curb Weight,
# Engine Size, Highway MPG
###############################

# Create Linear Regression Model
lm3 = LinearRegression()

# Equation given by Yhat = a + b1X1 + b2X2 + b3X3 + b4X4

# Develop model using these variables as predictor variables
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

# Fit linear model using four above-mentioned variables
lm3.fit(Z, df['price'])

# Intercept value
print(lm3.intercept_)

# Slope value
print(lm3.coef_)

# Equation:
# Price = -15678.742628061467 + 52.65851272 x horsepower + 4.69878948 x curb-weight + 81.95906216 x engine-size + 33.58258185 x highway-mpg

###############################
# Price, Normalised Losses,
# Highway MPG
###############################

# Create Linear Regression Model
lm4 = LinearRegression()

# Equation given by Yhat = a + b1X1 + b2X2 + b3X3

# Develop model using these variables as predictor variables
Z2 = df[['normalized-losses','highway-mpg']]

# Fit linear model using above variables
lm4.fit(Z2, df['price'])

# Intercept value
print(lm4.intercept_)

# Slope value
print(lm4.coef_)

#####################################################
# Model Evaluation Using Visualisation
#####################################################

###############################
# Regression Plot
###############################

# Plot of Highway MPG
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)


# Plot of Peak RPM
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)