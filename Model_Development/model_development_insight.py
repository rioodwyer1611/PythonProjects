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

Y_hat = lm3.predict(Z)
plt.figure(figsize=(width, height))


ax1 = sns.kdeplot(df['price'], color="r", label="Actual Value")
sns.kdeplot(Y_hat, color="b", label="Fitted Values" , ax=ax1)


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

###############################
# Polynomial Transformations
###############################

from sklearn.preprocessing import PolynomialFeatures

pr=PolynomialFeatures(degree=2)
print(pr)

Z_pr=pr.fit_transform(Z)

# In the original data, there are 201 samples and 4 features.
print(Z.shape)

# After the transformation, there are 201 samples and 15 features.
print(Z_pr.shape)

###############################
# Pipeline
###############################

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create a pipeline with a list of tuples with the model name and function.
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

pipe=Pipeline(Input)
print(pipe)

# Convert to float to avoid conversion warnings due to Standard Scaler only taking floats.
Z.astype(float)

# Can normalise data, perform a transformation, and fit model simulateously.
pipe.fit(Z,y)

# Can normalise data, perform a transformation, and create a prediction.
ypipe=pipe.predict(Z)
print(ypipe[0:4])

# Pipeline that standardizes the data, then produce a prediction using a linear regression model using the features Z and target y.
Input2 = [('scale', StandardScaler()), ('model', LinearRegression())]

pipe2 = Pipeline(Input2)
print(pipe2)

pipe2.fit(Z,y)

ypipe2 = pipe2.predict(Z)
print(ypipe2[0:10])

#####################################################
# Measures for In-Sample Evaluation
#####################################################

# Two types:
# - R-squared
# - Mean Square Error (MSE)

###############################
# Model 1: SLR
###############################

# Highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))

# Finding the MSE

from sklearn.metrics import mean_squared_error

Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])

mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

###############################
# Model 2: MLR
###############################

# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))

# Finding the MSE
Y_predict_multifit = lm.predict(Z)

print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

###############################
# Model 3: Polynomial Fit
###############################

from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)

# Calculate MSE

print(mean_squared_error(df['price'], p(x)))

#####################################################
# Prediction and Decision Making
#####################################################

###############################
# Prediction
###############################

import matplotlib.pyplot as plt
import numpy as np

# Create new input
new_input=np.arange(1, 100, 1).reshape(-1, 1)

# Fit model
lm.fit(X,Y)
print(lm)

# Produce Prediction
yhat=lm.predict(new_input)
print(yhat[0:5])

# Plot Data
plt.plot(new_input, yhat)
plt.show()

#####################################################
# What makes a good model?
#####################################################

# Closer to 1 R^2 value.
# Lower MSE value.