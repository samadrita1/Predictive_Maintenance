import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.metrics import roc_curve, auc


plt.rcParams['figure.figsize'] = (10.0, 5.0)

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
    
#Visualize
import matplotlib.pyplot as plt
df1[[3]].hist(figsize=(12, 4));
df1[[18]].plot(kind='density', subplots=True,layout=(1, 2), sharex=False, figsize=(12, 4));

#AFTER DECIDING MODEL
clf=RandomForestClassifier(random_state=0)
p=np.size(df1.iloc[0,:])
X=df1.iloc[:,3:p-2].values
y=df1.iloc[:,p-1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

X1_train=X_train
X1_test=X_test

##################################################
#Customized backward elimination
j=np.size(X_train[0,:])
clf.fit(X1_train, y_train)
y_pred = clf.predict(X1_test)
cm1 = confusion_matrix(y_test, y_pred)
g=cm1[1][1]/(cm1[1][1]+cm1[1][0]) #storing the recall value with all features
while(j>7):
    sen_arr=[]
    s=np.size(X_train[0,:])
    for i in range(s):
        X1_train = np.delete(X_train, i, 1)
        X1_test = np.delete(X_test, i, 1)
        clf.fit(X1_train, y_train)
        y_pred = clf.predict(X1_test)
        cm1 = confusion_matrix(y_test, y_pred)
        sen_arr.insert(i,cm1[1][1]/(cm1[1][1]+cm1[1][0]))
    m=sen_arr.index(max(sen_arr))
    print(m)
    print(sen_arr)
    if (g<sen_arr[m]):
        g=sen_arr[m]
        X_train = np.delete(X_train, m,1)
        X_test = np.delete(X_test, m, 1)
        print("Deleted")
    else:
        break
    j=j-1
#################################################
#Applying Recursive Feature Elimination
model = RandomForestClassifier()
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

clf=RandomForestClassifier(random_state=0)

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

param_grid = {"n_estimators": np.arange(350, 550, 5),
              "max_depth": np.arange(20, 100, 2),
              "min_samples_split": np.arange(2,100,2),
              "max_leaf_nodes": np.arange(100,150,1),
              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}
             
index = 1
for parameter, param_range in dict.items(param_grid):   
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
        
param_grid2 = {"n_estimators": [453,480],
              "max_depth": [25,70],
              "min_samples_split": [2,3,4],
              "criterion":['entropy','gini'],
              "min_weight_fraction_leaf": [0.0]}
              
grid_search = GridSearchCV(clf, param_grid=param_grid2)
grid_search.fit(X_train, y_train)
cl=grid_search.best_estimator_

report(grid_search.grid_scores_, 4)
'''
Model with rank: 1
Mean validation score: 0.8737)
Parameters: {'n_estimators': 360, 'min_samples_split': 2, 'criterion': 'entropy', 'max_depth': 70, 'min_samples_leaf': 1}
'''   
#manual tweaking one-by-one
param_grid3 = {"n_estimators": [525],
              "max_depth": [70],
              "min_samples_split": [2],
              "criterion":['gini','entropy'],
              "min_weight_fraction_leaf": [0.0]}
         
#n_estimators =400, min_samples_split= 10, min_samples_leaf=1, max_depth= 70, criterion = 'entropy', random_state = 0, n_jobs = -1, max_features=0.99
grid_search = GridSearchCV(clf, param_grid=param_grid3)
grid_search.fit(X_train, y_train)
cl1=grid_search.best_estimator_

report(grid_search.grid_scores_, 4)

# Fitting RandomForest to the Training set (Tuned values)
cl=RandomForestClassifier(n_estimators=485,max_depth=100,min_samples_split=2,class_weight={0:.30, 1:.70},
                          criterion='entropy',max_leaf_nodes=205,random_state=0,max_features=0.99)
cl.fit(X_train, y_train)

# Predicting the Test set results
y_pred = cl.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

#Testing on entire data (train plus test)
X_test1=df1.iloc[:,3:p-2].values
y_test1=df1.iloc[:,p-1].values

cl.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = cl.predict(X_test1)

# Making the Confusion Matrix
cm1 = confusion_matrix(y_test1, y_pred1)

#PERFORMANCE MEASURES
#ROC Curve

# calculate the fpr and tpr for all thresholds of the classification
probs = cl.predict_proba(X_test)
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