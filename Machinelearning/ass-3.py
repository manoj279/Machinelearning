from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve,auc
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('fraud_data.csv')


def answer_one():
    z=df['Class'].values
    sz=z.size
    t=np.count_nonzero(z==1)
    return float(t)/float(sz)
answer_one()

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def answer_two():
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    y_dummy_predictions = dummy_majority.predict(X_test)
    a=accuracy_score(y_test,y_dummy_predictions)
    rs=recall_score(y_test,y_dummy_predictions)
    return (a,rs)
answer_two()

def answer_three():
    svm = SVC().fit(X_train, y_train)
    ypred=svm.predict(X_test)
    a=accuracy_score(y_test,ypred)
    rs=recall_score(y_test,ypred)
    ps=precision_score(y_test,ypred)
    return(a,rs,ps)
answer_three()

def answer_four():
    svm = SVC(C=1e9,gamma=1e-07).fit(X_train, y_train)
    score=svm.decision_function(X_test)
    l=score>-220
    con=confusion_matrix(y_test,l)
    return con
answer_four()

def answer_five():
    lr = LogisticRegression().fit(X_train, y_train)
    lr_predicted = lr.predict_proba(X_test)[:,1]


    precision, recall, thresholds = precision_recall_curve(y_test,lr_predicted)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]
    # #
    # plt.figure()
    # plt.xlim([0.0, 1.01])
    # plt.ylim([0.0, 1.01])
    # plt.plot(precision, recall, label='Precision-Recall Curve')
    # plt.plot(0.75, 0.838, 'o', markersize = 0.2, fillstyle = 'none', c='r', mew=3)
    # plt.xlabel('Precision', fontsize=16)
    # plt.ylabel('Recall', fontsize=16)
    # plt.axes().set_aspect('equal')
    # plt.show()
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_predicted)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    # plt.figure()
    # plt.xlim([-0.01, 1.00])
    # plt.ylim([-0.01, 1.01])
    # plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    # plt.xlabel('False Positive Rate', fontsize=16)
    # plt.ylabel('True Positive Rate', fontsize=16)
    # plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
    # plt.legend(loc='lower right', fontsize=13)
    # plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    # plt.plot(0.16, 0.824, 'o', markersize = 0.2, fillstyle = 'none', c='r', mew=3)
    # plt.axes().set_aspect('equal')
    # plt.show()
    recall_query = recall[np.argmin(abs(precision - 0.75))]
    tpr_query = tpr_lr[np.argmin(abs(fpr_lr - 0.16))]

    return (recall_query, tpr_query)

answer_five()
