import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.metrics import roc_curve, auc


plt.rcParams['figure.figsize'] = (20.0, 10.0)

df1=pd.read_csv("Final_HD.csv")
k=df1.index.values
a=[i for i in k]
k=0
p=np.size(df1.iloc[0,:])
for i in df1.iloc[:,0]:
    if(df1.iloc[k,p-2:p-1].values[0] <= 240):
        df1.at[a[k],p-1:p] =1
    else:
        df1.at[a[k],p-1:p] =0
    k=k+1
    
#AFTER DECIDING MODEL

from sklearn.linear_model import LogisticRegression
cl = LogisticRegression(random_state=0)

#Applying Recursive Feature Elimination
p=np.size(df1.iloc[0,:])
X=df1.iloc[:,4:p-2].values
y=df1.iloc[:,p-1].values
model = LogisticRegression()
rfe = RFE(model, 5)
selector = rfe.fit(X, y)
arr=selector.ranking_

k=4  #since SMART 9 raw data is has been replaced by RUL
j=0
f=[]
for i in arr:
    if (i==1):
        f.insert(j,k)
        j=j+1
    k=k+1
f.insert(0,0)
f.insert(1,1)
f.insert(2,2)
f.insert(k,p-2)
f.insert(k+1,p-1)

df1=pd.DataFrame(df1.iloc[:,f])

#Applying classification model
p=np.size(df1.iloc[0,:])
X=df1.iloc[:,3:p-2].values
y=df1.iloc[:,p-1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#parameter selection
# Create regularization penalty space
penalty = ['l1', 'l2']
# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
# Create grid search using 5-fold cross validation
clf = GridSearchCV(cl, hyperparameters, cv=5, verbose=0)
# Fit grid search
best_model = clf.fit(X_train, y_train)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

#restart

clf= LogisticRegression(random_state=0)
def evaluate_param(parameter, num_range, index):
    grid_search = GridSearchCV(clf, param_grid = {parameter: num_range})
    grid_search.fit(X_train, y_train)
    
    df = {}
    for i, score in enumerate(grid_search.grid_scores_):
        df[score[0][parameter]] = score[1]
       
    
    df = pd.DataFrame.from_dict(df, orient='index')
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by='index')
 
    plt.subplot(3,2,index)
    plot = plt.plot(df['index'], df[0])
    plt.title(parameter)
    plt.show()
    return plot, df

param_grid1 = {"C": np.logspace(0, 4, 10),
               "max_iter": np.arange(100, 150, 5)}
             
index = 1
for parameter, param_range in dict.items(param_grid1):   
    evaluate_param(parameter, param_range, index)
    index += 1

# Utility function to report best scores
def report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
        
param_grid1a = {"penalty": ['l1'],
              "C": [1, 1000],
              "solver": ['liblinear','saga']}

param_grid2a = {"penalty": ['l2'],
              "C": [1,1000],
              "solver": ['newton-cg','lbfgs','sag'],
              "max_iter": np.arange(100,150,5)}
              
grid_search = GridSearchCV(clf, param_grid=param_grid1a)
grid_search.fit(X_train, y_train)
cl=grid_search.best_estimator_

report(grid_search.grid_scores_, 4)

grid_search = GridSearchCV(clf, param_grid=param_grid2a)
grid_search.fit(X_train, y_train)
cl=grid_search.best_estimator_

report(grid_search.grid_scores_, 4)

#manual tweaking one-by-one
param_grid3 = {"penalty": ['l1'],
               "C": [1, 500, 1000],
               "solver": ['liblinear']}
         
#n_estimators =400, min_samples_split= 10, min_samples_leaf=1, max_depth= 70, criterion = 'entropy', random_state = 0, n_jobs = -1, max_features=0.99
grid_search = GridSearchCV(clf, param_grid=param_grid3)
grid_search.fit(X_train, y_train)
cl1=grid_search.best_estimator_

report(grid_search.grid_scores_, 4)

#model
cl=LogisticRegression(penalty='l2',random_state=0,class_weight={0:.35, 1:.65}, C=1, solver='newton-cg', max_iter=700)
cl.fit(X_train, y_train)

# Predicting the Test set results
y_pred = cl.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

#Testing on entire data (train plus test)
X_test1=df1.iloc[:,3:p-2].values
y_test1=df1.iloc[:,p-1].values

clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = clf.predict(X_test1)

# Making the Confusion Matrix
cm1 = confusion_matrix(y_test1, y_pred1)

#PERFORMANCE MEASURES
#ROC Curve

# calculate the fpr and tpr for all thresholds of the classification
probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

accuracy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
misclassification_rate=(cm[0][1]+cm[1][0])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
precision=cm[1][1]/(cm[1][1]+cm[0][1])
Sensitivity=cm[1][1]/(cm[1][1]+cm[1][0])
Specificity=cm[0][0]/(cm[0][0]+cm[0][1])
F1_Score = 2*((precision*Sensitivity)/(precision+Sensitivity))