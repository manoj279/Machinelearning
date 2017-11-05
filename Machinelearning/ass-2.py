import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.regression import r2_score
from sklearn.tree import DecisionTreeClassifier
#from adspy_shared_utilities import plot_feature_importances
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);


# NOTE: Uncomment the function below to visualize the data, but be sure
# to **re-comment it before submitting this assignment to the autograder**.
#part1_scatter()

x=x.reshape(-1,1)
def answer_one():
    a=[]
    for i in [1,3,6,9]:
        poly = PolynomialFeatures(degree=i)
        X_poly=poly.fit_transform(x)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y,
                                                           random_state = 0)
        linreg = LinearRegression().fit(X_train, y_train)

        pred_data=np.linspace(0,10,100)
        pred_data=pred_data.reshape(-1,1)
        pred=linreg.predict(poly.fit_transform((pred_data)))
        a.append(pred)
    a=np.array(a)
    return a

answer_one()

# feel free to use the function plot_one() to replicate the figure
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    plt.show()

#plot_one(answer_one())

def answer_two():
    r2_train=[]
    r2_test=[]
    for i in range(0,10):
        poly = PolynomialFeatures(degree=i)
        X_poly=poly.fit_transform(x)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y,
                                                           random_state = 0)
        linreg = LinearRegression().fit(X_train, y_train)
        r2_train.append(linreg.score(X_train,y_train))
        r2_test.append(linreg.score(X_test,y_test))
    r2_test=np.array(r2_test)
    r2_train=np.array(r2_train)
    return (r2_train,r2_test)
answer_two()

def answer_three():
    r2_train,r2_test=answer_two()
    return (3,9,7)
answer_three()

def answer_four():
    poly = PolynomialFeatures(degree=12)
    X_poly=poly.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y,random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
    r2_test1=linreg.score(X_test,y_test)

    poly = PolynomialFeatures(degree=12)
    X_poly=poly.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y,random_state = 0)
    linreg1 = Lasso(alpha=0.01,max_iter=10000).fit(X_train,y_train)
    r2_test2=linreg1.score(X_test,y_test)
    return (r2_test1,r2_test2)
answer_four()

mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

def answer_five():
    clf=DecisionTreeClassifier(random_state=0).fit(X_train2,y_train2)
    importance = pd.Series(clf.feature_importances_, index=X_train2.columns)
    importance=importance.sort_values(ascending=False)
    a=importance.iloc[0:5].index.tolist()
    return a
answer_five()

def answer_six():
    param_range = np.logspace(-4, 1, 6)
    train_scores, test_scores = validation_curve(SVC(kernel='rbf',C=1), X_subset, y_subset,param_name='gamma',param_range=param_range, cv=3)
    train_scores=np.mean(train_scores,axis=1)
    test_scores=np.mean(test_scores,axis=1)
    return (train_scores,test_scores)
answer_six()