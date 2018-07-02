import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import SVC

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
'''
# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
p=np.size(df1.iloc[0,:])
# load data
X1 = df1.iloc[:,3:p-2]
y1 = df1.iloc[:,p-1]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X1, y1)
print(model.feature_importances_)
d=pd.DataFrame()
d['col_name'] = pd.Series(np.arange(3,18,1))
d['importance'] = pd.Series(model.feature_importances_)
d=d.sort_values(by=['importance'], ascending=[False])
#selecting 5 most important features
b=[]
b.insert(0,0)
b.insert(1,1)
b.insert(2,2)
k=3
for i in range(3,8):
    b.insert(k,d.iloc[k-3,0])
    k=k+1
b.insert(8,p-2)
b.insert(9,p-1)
b=list(map(int, b))
#Updating dataset with top 5 features
df1=pd.DataFrame(df1.iloc[:,b])
'''

clf = SVC(random_state=0)
#Load data
p=np.size(df1.iloc[0,:])
X=df1.iloc[:,3:p-2].values
y=df1.iloc[:,p-1].values
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
#Feature selection
sen=cm[1][1]/(cm[1][1]+cm[1][0])

X1_train=X_train
X1_test=X_test
##################################################
#Customized backward elimination
j=np.size(X_train[0,:])
g=sen
while(j>2):
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
##################################################
#parameter selection
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

param_grid1 = {"C": [1,2],
               "gamma":np.arange(0.05,0.85,0.05),
               "coef0":[0.0, 0.02],
               "tol":[0.001, 0.0005]}
             
index = 1
for parameter, param_range in dict.items(param_grid1):   
    evaluate_param(parameter, param_range, index)
    index += 1
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
f1=[0,1,2,4,6,7,8,10,11,12,15,17,18,19] #Experiment with RFE features
check1=pd.DataFrame(df1.iloc[:,f1])
p=np.size(check1.iloc[0,:])
X=check1.iloc[:,3:p-2].values
y=check1.iloc[:,p-1].values
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#model
cl = SVC(kernel = 'rbf', class_weight={0:.30, 1:.70}, random_state = 0, C=3, gamma=0.15) #assigned class weights depending on ration of true 0s and true 1s
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

accuracy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
misclassification_rate=(cm[0][1]+cm[1][0])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
precision=cm[1][1]/(cm[1][1]+cm[0][1])
Sensitivity=cm[1][1]/(cm[1][1]+cm[1][0])
Specificity=cm[0][0]/(cm[0][0]+cm[0][1])
F1_Score = 2*((precision*Sensitivity)/(precision+Sensitivity))