import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from scipy import interpolate, interp
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Read csv file (Testing with few rows first)
dataset=pd.read_csv("HD.csv")

#FEATURE ENGINEERING
#delete 1st column:(unrequired data)
dataset=pd.DataFrame(dataset.iloc[:, 1:])

#Accept the normalized values in the final dataset
a=[0,1,2,3,19,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92]
df=pd.DataFrame(dataset.iloc[:, a])

#failed devices
f_devices= pd.DataFrame(df.iloc[:,0][df.iloc[:,3]==1])

#Store serial numbers of the products which show failure
a=f_devices.iloc[:,0].values


#Create datasets for each device (only devices showing failure at some point)
index = [i for i in a]
dataframe_collection = {}
df1=pd.DataFrame()
for i in index:
    new_data = df[df.iloc[:,0]==i]
    new_data=new_data.sort_values(by=['smart_9_raw'], ascending=[True])
    dataframe_collection[i] = new_data
    df1=df1.append(new_data)
#df1 = df1.reset_index(drop=True)

k=5
p=np.size(df1.iloc[0,:])
b=[] #to store columns with all nan value
for i in range(5,p):
    t=df1.iloc[:,i].values
    if (np.isnan(t).all()==False):#no change required
        b.insert(k,i)
        k=k+1
b.insert(0,0)
b.insert(1,1)
b.insert(2,2)
b.insert(3,3)
b.insert(4,4)

df1=pd.DataFrame(df1.iloc[:,b])

df1=df1.interpolate() #To interpolate

#Demonstrate working of extrapolation
x = [101, 102, 103, 104, 105,106, 107, 109, 111, 113]
y = [0,0,0,0,0,0,0,0,1,1]
f = interpolate.interp1d(x, y, fill_value='extrapolate')
xnew = [95, 96, 97, 98, 99, 100, 101, 114, 115, 117]
ynew = f(xnew)   # use interpolation function returned by `interp1d`
interp([114,100,110,112,108], x, y)
#Application of extrapolation
p=np.size(df1.iloc[0,:])
for i in range(5,p):
    n=df1.iloc[:,i].isnull().sum()
    if (n>0 and n<5642):
        check2=interpolate.interp1d(df1.iloc[n:,4],df1.iloc[n:,i],fill_value='extrapolate')
        check3=check2(df1.iloc[:n,4])
        df1.iloc[:n,i]=check3

#Visualize to detect relationship
df1.hist() #shows frequency of the values of every feature
plt.show()

p=np.size(df2.iloc[0,:])   #To check if linear interpolation is the correct choice
x1=np.asarray(df1.iloc[:,4])
x=np.asarray(df2.iloc[:,4])
for i in range(5, p):
    y=np.asarray(df2.iloc[:,i])         #line 87 and 88 helps to identify the new values after interpolation (blue)
    y1=np.asarray(df1.iloc[:,i])     
    plt.scatter(x, y, color='blue')
    plt.scatter(x1, y1, color='red')
    plt.xlabel('time')
    plt.ylabel('feature')
    plt.show()

'''
k=2
b=[] #to store columns with no nan value
for i in range(2,50):
    t=df1.iloc[:,i].values
    if (np.isnan(t).all()==False):#no change required
        b.insert(k,i)
        k=k+1
b.insert(0,0)
b.insert(1,1)

X=df1.iloc[0:3500,6]
x_plot=df1.iloc[3500:5642,6]
y=df1.iloc[0:3500,4:5]
y_true=df1.iloc[3500:5642,4:5]
colors = ['teal', 'yellowgreen', 'gold']
lw = 0.5
for count, degree in enumerate([3]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(y, X)
    x_plot = model.predict(y_true)
    plt.plot(y_true, x_plot, color=colors[count], linewidth=lw,
             label="degree %d" % degree)

plt.legend(loc='lower left')
df1.iloc[0:50,6].interpolate('index').plot(marker='o')
df1.iloc[0:50,6:12].interpolate('index')
plt.show()
s = pd.Series([np.nan, np.nan,np.nan,1, np.nan, np.nan, 0, np.nan, 3])
s.interpolate(limit=10,limit_direction='both')

plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()

df1=pd.DataFrame(df1.iloc[:,b])
p=np.size(df1.iloc[0,:])

np.isnan(df1.iloc[:,features[0]]).any()

for i in range(4,p):
    features = [6]
    filtered_data = df1.iloc[:,features[0]][~np.isnan(df1.iloc[:,features[0]])]
    plt.hist(filtered_data, bins=20, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(filtered_data.mean(), color='black', linestyle='dashed', linewidth=3)
    plt.axvline(filtered_data.median(), color='red', linestyle='dashed', linewidth=3)
    plt.show()

df1[[12]].hist(figsize=(12, 4));
df1[features].plot(kind='density', subplots=True,layout=(1, 2), sharex=False, figsize=(12, 4));
#Saving data before nan replacement
df_a=pd.DataFrame(df1.iloc[:,[0,1,3]])

#Replacing all NaN values with mean of same class
df_b = df1.groupby("failure").transform(lambda x: x.fillna(x.mean()))

#Joining nan replaced set with stored features
df1=df_a.join(df_b)
'''

#Detect columns still with all NaN values and update dataset without them
k=5
p=np.size(df1.iloc[0,:])
b=[] #to store columns with all nan value
for i in range(5,p):
    t=df1.iloc[:,i].values
    if (np.isnan(t).all()==False):
        b.insert(k,i)
        k=k+1
b.insert(0,0)
b.insert(1,1)
b.insert(2,2)
b.insert(3,3)
b.insert(4,4)

df1=pd.DataFrame(df1.iloc[:,b])

#Check Variance of each attribute and eliminate features with 0.0 variance
test=df1.var()
c=[]
k=2
for i in range(1,np.size(test)):
    if ((test[i]==float(0))==False):
        c.insert(k,i+2)
        k=k+1
c.insert(0,0)
c.insert(1,1)

df1=pd.DataFrame(df1.iloc[:,c])

#Visualize relationship of features with target variable
plt.rcParams['figure.figsize'] = (10, 20)
y1=df1.iloc[:,2].values
x1=df1.iloc[:,4].values
plt.scatter(y1,x1,color='red')

#Observing the mutual information (identifies any form of dependency with TIME(SMART 9) variable)
y=df1.iloc[:,3].values
k=4
e=[]
p=np.size(df1.iloc[0,:])
for i in range(0,p-4):
    X=df1.iloc[:,i+4:i+5].values
    mi = mutual_info_regression(X, y, random_state=1)  #Computationally expensive
    print(mi[0])
    if ((mi[0]==0)==False):
        e.insert(k,i+4)
        k=k+1
e.insert(0,0)
e.insert(1,1)
e.insert(2,2)
e.insert(3,3)

df1=pd.DataFrame(df1.iloc[:,e])

#Add RUL column (based on SMART 9)
m={}
for i in index:
    m[i]=dataframe_collection[i]['smart_9_raw'].max()
    
df1=df1.sort_values(by=['serial_number', 'smart_9_raw'], ascending=[True,True])
df1['RUL'] =np.nan
k=df1.index.values
a=[i for i in k]
k=0
p=np.size(df1.iloc[0,:])
for i in df1.iloc[:,0]:
    df1.at[a[k],p-1:p]=m[i]-df1.iloc[k:k+1,3:4].values[0][0]
    k=k+1
    
#Add column to label all rows within failing time window as 1
#Failure time window: 10 days ('RUL' <= 240 hours)
df1['Label'] =np.nan
k=0
p=np.size(df1.iloc[0,:])
for i in df1.iloc[:,0]:
    if(df1.iloc[k,p-2:p-1].values[0] <= 240):
        df1.at[a[k],p-1:p] =1
    else:
        df1.at[a[k],p-1:p] =0
    k=k+1
    
#AFTER DECIDING MODEL

#Applying Recursive Feature Elimination
p=np.size(df1.iloc[0,:])
X=df1.iloc[:,4:p-2].values
y=df1.iloc[:,p-1].values
model = RandomForestClassifier()
rfe = RFE(model, 15)
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

#Storing feature engineered dataframe in a csv file to avoid repeated processing        
df1.to_csv('Final_HD.csv', sep=',', encoding='utf-8',index=False)
df1=pd.read_csv("Final_HD.csv")

#Applying classification model
p=np.size(df1.iloc[0,:])
X=df1.iloc[:,3:p-2].values
y=df1.iloc[:,p-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#RANDOM SEARCH to narrow down possibilities
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 300, stop = 1500, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
#Criterion
criterion = ['entropy','gini']
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'criterion':criterion}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random.best_params_

#GRID SEARCH
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [60, 70, 80],
    'max_features':[0.5,0.75,0.99],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [350,400,450],
    'criterion': ['entropy']
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,cv = 3, n_jobs = -1, verbose = 2)
                      
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_

best_grid = grid_search.best_estimator_
best_grid.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = best_grid.predict(X_test)

# Making the Confusion Matrix
cm1 = confusion_matrix(y_test, y_pred1)

# Fitting RandomForest to the Training set (Tuned values)
classifier = RandomForestClassifier(n_estimators =400, min_samples_split= 10, min_samples_leaf=1, max_depth= 70, criterion = 'entropy', random_state = 0, n_jobs = -1, max_features=0.99)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)