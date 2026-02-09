# Continuation of Model Development Work:
# pip install pandas matplotlib scipy scikit-learn seaborns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

#####################################################
# Set Up Data
#####################################################

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv", header=0)

print(df.head())

df = df._get_numeric_data()
print(df.head())
# Remove noisy columns
df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)
print(df.head())


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    plt.show()

#####################################################
# Training and Testing
#####################################################

###############################
# Split into Training and
# Testing Data
###############################

y_data = df['price']
# Drop price from X Data
x_data = df.drop('price', axis = 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

# Split up the dataset such that 40% of the data samples will be utilized for testing. 
# Set the parameter "random_state" equal to zero. 
# The output of the function should be the following: "x_train1" , "x_test1", "y_train1" and "y_test1".

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4)
print("number of test samples :", x_test1.shape[0])
print("number of training samples:", x_train1.shape[0])

from sklearn.linear_model import LinearRegression

# Create Linear Regression Object.
lre = LinearRegression()

# Fit model using 'horsepower'
lre.fit(x_train[['horsepower']], y_train)

# Calculate R^2 value of test values
print(lre.score(x_test[['horsepower']], y_test))

# Calculate R^2 value of training data
print(lre.score(x_train[['horsepower']], y_train))

# This shows that the R^2 value of the test data is much smaller than that of the training data.

###############################
# Cross-Validation Score
###############################

from sklearn.model_selection import cross_val_score

# Input as object, feature ('horsepower') and target data (y_data).
# CV determines the number of folds
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)

print(Rcross)

# Can produce mean of the folds and standard deviation.
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

# Can use the negative squared error as a parameter by setting scoring to 'neg-squared-error'
print("The negative squared error is ", -1 * cross_val_score(lre,x_data[['horsepower']], y_data, cv=4, scoring='neg_mean_squared_error'))

# Average R^2 using two folds, then find the average R^2 for the second fold utilizing the "horsepower" feature.
Rc = cross_val_score(lre, x_data[['horsepower']], y_data, cv=2)
print(Rc.mean())

###############################
# Cross-Validation Predict
###############################

from sklearn.model_selection import cross_val_predict

# Formatted as input object, feature ('horsepower'), and target data (y_data).
# CV determines number of folds.

yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
print(yhat[0:5])

#####################################################
# Overfitting, Underfitting and Model Selection
#####################################################

###############################
# Using an MLR object
###############################

# Create and train an MLR using horsepower, curb weight, engine size and highway mpg as features.
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

# Create a prediction using training data
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_train[0:5])

# Create a prediction using test data
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_test[0:5])

###############################
# Evaluation
###############################

# Figure 1: Predicted values using the training data compared to the actual values of the training data.
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

# Figure 2: Predicted value using the test data compared to the actual values of the test data.
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

# Outcome:
# Figures show that figure 1 fits the model far greater than figure 2.
# Should now check if polynomial matches the model any better.

from sklearn.preprocessing import PolynomialFeatures

###############################
# Overfitting
###############################

# Use 55% of data for training and the rest for testing
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

# Perform a 5 degree polynomial transformation
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
print(pr)

# Create Linear Regression Model and train it.
poly = LinearRegression()
poly.fit(x_train_pr, y_train)

yhat = poly.predict(x_test_pr)
print(yhat[0:5])

# Look at first 5 predict values and compare to actual targets.
print("Predicted Values:", yhat[0:4])
print("True Values:", y_test[0:4])

# Use PollyPlot function from before to display the training, testing and predicted data.

# Figure 3: A Polynomial Regression model: 
# Red dots represent training data, 
# Green dots represent test data, 
# Blue line represents the model prediction.

print(PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly, pr))

# Get R^2 of training data
poly.score(x_train_pr, y_train)

# Get R^2 of test data
poly.score(x_test_pr, y_test)

# See how R^2 changes on the test data for different polynomials.
# Plot results.
Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')
plt.show()

# This showed that the R^2 value steadily increased, peaking at an order 3 polynomial and then plummeted when an order 4 was used.

# Usable to any level of order by calling function.
def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)

# This function shows ever possible combination of order and test_data graphed.
# Would not recommend to uncomment without purpose.
#for order in range(0, 7):
#    for test_data in np.arange(0.05, 1.0, 0.05):
#        f(order=order, test_data=test_data)

#####################################################
# Ridge Regression
#####################################################

# Two degree polynomial transformation of data.
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

from sklearn.linear_model import Ridge

RidgeModel = Ridge(alpha=1)

# Can fit model using fit, like a normal model.
RidgeModel.fit(x_train_pr, y_train)

# Can also obtain a prediction.
yhat = RidgeModel.predict(x_train_pr)

# Compare first 5 instances of predictions.
print('Predicted:', yhat[0:4])
print('Test Set:', y_test[0:4].values)

# Select value of alpha that minimises test error
from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

# Iterates through all given alphas and appends them to a list.
for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

# Can plot following alpha data to see which is best.
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()

# Perform Ridge regression. 
# Calculate the R^2 using the polynomial features, 
# use the training data to train the model and use the test data to test the model. 
# The parameter alpha should be set to 10.

rge = Ridge(alpha=10)
rge.fit(x_train_pr, y_train)
rge.score(x_train_pr, y_train)


#####################################################
# Grid Search
#####################################################

from sklearn.model_selection import GridSearchCV

parameters1 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 1000000]}]

# Create Ridge Regression Object.
RR = Ridge()

# Create Ridge Grid Search Object.
Grid1 = GridSearchCV(RR, parameters1, cv=4)

# Fit Model.
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

# Object can find best parameter values on the validation data.
# Can obtain the estimator with the best parameters.
BestRR = Grid1.best_estimator_

# Can now test model on the data.
print(BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test))

# Can find best alpha.
best_alpha = Grid1.best_params_['alpha']
print(best_alpha)