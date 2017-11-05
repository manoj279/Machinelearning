from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import cm,pyplot

cancer = load_breast_cancer()

columns =['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness', 'mean compactness', 'mean concavity',
'mean concave points', 'mean symmetry', 'mean fractal dimension',
'radius error', 'texture error', 'perimeter error', 'area error',
'smoothness error', 'compactness error', 'concavity error',
'concave points error', 'symmetry error', 'fractal dimension error',
'worst radius', 'worst texture', 'worst perimeter', 'worst area',
'worst smoothness', 'worst compactness', 'worst concavity',
'worst concave points', 'worst symmetry', 'worst fractal dimension',
'target']
index =pd.RangeIndex(start=0, stop=569, step=1)

def answer_one():
    data = np.c_[cancer.data, cancer.target]
    columns = np.append(cancer.feature_names, ["target"])
    return pd.DataFrame(data, columns=columns)

def answer_two():
    cancerdf = answer_one()
    x,y=cancerdf.target.value_counts()
    s=Series([y,x],index=['malignant', 'benign'])
    return s

def answer_three():
    cancerdf=answer_one()
    X = cancerdf[cancerdf.columns[:-1]]
    y = cancerdf.target
    return X,y

def answer_four():
    X, y = answer_three()
    return train_test_split(X, y,train_size=426, test_size=143, random_state=0)

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train, y_train)
    return model

def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    knn=answer_five()
    return knn.predict(means)

def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    return knn.predict(X_test)

def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    return knn.score(X_test,y_test)


answer_one()
answer_two()
answer_three()
answer_four()
answer_five()
answer_six()
answer_seven()
answer_eight()