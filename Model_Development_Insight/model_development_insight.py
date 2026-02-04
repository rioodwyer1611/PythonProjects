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

#This plot will show a combination of a scattered data points (a scatterplot), 
# as well as the fitted linear regression line going through the data. This 
# will give us a reasonable estimate of the relationship between the two variables, 
# the strength of the correlation, as well as the direction 
# (positive or negative correlation).

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

plt.show()

# The variable "highway-mpg" has a stronger correlation with "price", it is approximate -0.704692  compared to "peak-rpm" which is approximate -0.101616. You can verify it using the following command:
df[["peak-rpm","highway-mpg","price"]].corr()

###############################
# Residual Plot
###############################

# Residual:
# The difference between the observed value (y) and the predicted value (Yhat) 
# is called the residual (e). When we look at a regression plot, the residual 
# is the distance from the data point to the fitted regression line.

# Residual Plot:
# A residual plot is a graph that shows the residuals on the vertical y-axis and 
# the independent variable on the horizontal x-axis.
# If the points in a residual plot are randomly spread out around the x-axis, 
# then a linear model is appropriate for the data.

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x=df['highway-mpg'], y=df['price'])
plt.show()

###############################
# Multiple Linear 
# Regression Plot
###############################

# Can't use a regression or residual plot to visualise MLR.
# Can use a distribution plot.

Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))


ax1 = sns.kdeplot(df['price'], hist=False, color="r", label="Actual Value")
sns.kdeplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()

###############################
# Polynomial Regressions
###############################

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-mpg']
y = df['price']

# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

PlotPolly(p, x, y, 'highway-mpg')

np.polyfit(x, y, 3)

# 11 Order Polynomial 
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1,x,y, 'Highway MPG')