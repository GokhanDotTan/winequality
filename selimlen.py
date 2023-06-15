
"""
Gökhan Tan
Created on Tue Jun 13 18:05:11 2023

Wine Quality RandomForestClassifier

"""

# STEP 1, after the dataset is loaded, necessary controls and directions are created and
# The step in which we will perform the removal of missing or unnecessary parts from the data set

# import startup libraries
import pandas as pd
import numpy as np

# datasetini yükle
wine = pd.read_csv('winequalityN.csv')
# get information about the dataset
wine.info()

# Let's check the values ​​of the features in the dataset. I found 1 sufficient as I am on step 1.
wine.quality.unique()

# Check for null values ​​in the dataset
wine.isnull().sum()

# Since there are very few zeros in the dataset, I decided to replace it with nulls when averaging the features.
wine.mean()
wine_mean = wine.fillna(wine.mean())

# check if null value remains
wine_mean.isnull().sum()
wine_mean.info()
# Check if wine and wine_mean averages are the same
print(wine.mean())
print(wine_mean.mean())
 
# Here, null values ​​are dropped and the white place is set to 1 and the red place to 0 in the wine type column.
wine1 = wine.dropna()
wine1['whitewine'] = [
    1 if typ == 'white' else 0 for typ in wine1['type']]
wine1.drop(['type'], axis=1, inplace=True)

# copy the dataset
def preprocess_inputs(df):
    df = df.copy()
    
    return df
df1 = preprocess_inputs(wine1)

# heatmap review
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,10))
sns.heatmap(df1.corr(),annot=True)
plt.show()

# comparison of quality values ​​(a circle diagram may also be preferred)
plot= sns.catplot(x='quality', data = df1, kind = 'count')

# looking for correlations with quality
plt.figure(figsize=(20,15))
df1.corr()['quality'].apply(lambda x: abs(x)).sort_values(ascending=False).iloc[1:12][::-1].plot(kind='barh') 
plt.title("correlated features", size=20, pad=26)
plt.xlabel("Correlation Coefficients")
plt.ylabel("Features")
plt.show()

# How does the quality change with the change in two different features (I made 9) but for detailed review
# This figure can be increased up to 20-30-40 pieces.
sns.set_style('whitegrid')
fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(15,10))
ax = ax.flatten()
sns.scatterplot(ax=ax[0],x='density',y='pH',data=wine1,hue='quality')
sns.scatterplot(ax=ax[1],x='density',y='alcohol',data=wine1,hue='quality')
sns.scatterplot(ax=ax[2],x='sulphates',y='alcohol',data=wine1,hue='quality')
sns.scatterplot(ax=ax[3],x='chlorides',y='alcohol',data=wine1,hue='quality')
sns.scatterplot(ax=ax[4],x='residual sugar',y='alcohol',data=wine1,hue='quality')
sns.scatterplot(ax=ax[5],x='citric acid',y='alcohol',data=wine1,hue='quality')
sns.scatterplot(ax=ax[6],x='fixed acidity',y='alcohol',data=wine1,hue='quality')
sns.scatterplot(ax=ax[7],x='free sulfur dioxide',y='alcohol',data=wine1,hue='quality')
sns.scatterplot(ax=ax[8],x='total sulfur dioxide',y='alcohol',data=wine1,hue='quality')
plt.show()

# separating my target and feature sets
X = df1.drop('quality', axis = 1).values
y = df1['quality']

# Model selection and model application
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
clf =RandomForestClassifier(n_estimators=800,max_features=10,random_state=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=13)

# Printing numeric numbers of train, target and test data on console, 20% of dataset is test set 
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# The part where we fit the module and output the accuracy value
clf= RandomForestClassifier()
clf= clf.fit(X_train,y_train)
y_pred =clf.predict(X_test)
print("Accuracy :" ,metrics.accuracy_score(y_test,y_pred))

# The part where we visualize the confusion matrix output and matrix values ​​in the heatmap
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
cm1 = confusion_matrix(y_test,y_pred)
import seaborn as sns
sns.heatmap(cm1,annot= True, fmt='d', cmap='rocket')
print(conf_mat)

        






































































































































    






































    
    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    