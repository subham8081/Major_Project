#IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from mlxtend.plotting import plot_decision_regions
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline

#READING DATASET FILE
diabetes_df = pd.read_csv(r"C:\Users\subha\OneDrive\Desktop\Major Project\Code\diabetes.csv", encoding="ISO-8859-1")
print(diabetes_df)

diabetes_df.head()
diabetes_df.columns
diabetes_df.info()
diabetes_df.describe()
diabetes_df.describe().T
diabetes_df.isnull().head(10)
diabetes_df.isnull().sum()
diabetes_df_copy = diabetes_df.copy(deep = True)
diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(diabetes_df_copy.isnull().sum())

#Data Visualization
#Plotting the data distribution plots before removing null values

p = diabetes_df.hist(figsize = (20,20))

#imputing the mean value of the column to each missing value of that particular column

diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].mean(), inplace = True)
diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].mean(), inplace = True)
diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace = True)
diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].median(), inplace = True)
diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace = True)

#Plotting the distributions after removing the NAN values.
p = diabetes_df_copy.hist(figsize = (20,20))

#Plotting Null Count Analysis Plot
p = msno.bar(diabetes_df)
color_wheel = {1: "#0392cf", 2: "#7bc043"}
colors = diabetes_df["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(diabetes_df.Outcome.value_counts())
p=diabetes_df.Outcome.value_counts().plot(kind="bar")
plt.subplot(121), sns.distplot(diabetes_df['Insulin'])
plt.subplot(122), diabetes_df['Insulin'].plot.box(figsize=(16,5))
plt.show()

#Correlation between all the features before cleaning
plt.figure(figsize=(12,10))

# seaborn has an easy method to showcase heatmap
p = sns.heatmap(diabetes_df.corr(), annot=True,cmap ='RdYlGn')

#Scaling the Data
diabetes_df_copy.head()
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes_df_copy.drop(["Outcome"],axis = 1),), columns=['Pregnancies', 
'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()

#Model Building
#Splitting the dataset
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

#Random Forest
#Building the model using RandomForest
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,random_state=7)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

#check the accuracy of the model on the training dataset
rfc_train = rfc.predict(X_train)

from sklearn import metrics
print("Accuracy_Score of Random Forest after Training =", format(metrics.accuracy_score(y_train, rfc_train)))

from sklearn import metrics
predictions = rfc.predict(X_test)
print("Accuracy_Score of Random Forest after Testing =", format(metrics.accuracy_score(y_test, predictions)))

#Decision Tree
#Building the model using DecisionTree
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
from sklearn import metrics

predictions = dtree.predict(X_test)
print("Accuracy Score of Decision Tree =", format(metrics.accuracy_score(y_test,predictions)))
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))

#XgBoost classifier
#Building model using XGBoost
from xgboost import XGBClassifier

xgb_model = XGBClassifier(gamma=0)
xgb_model.fit(X_train, y_train)

from sklearn import metrics

xgb_pred = xgb_model.predict(X_test)
print("Accuracy Score of XGBoost Classifier =", format(metrics.accuracy_score(y_test, xgb_pred)))

#Support Vector Machine (SVM)
#Building the model using Support Vector Machine (SVM)
from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)
from sklearn import metrics

print("Accuracy Score of SVM =", format(metrics.accuracy_score(y_test, svc_pred)))
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, svc_pred))
print(classification_report(y_test,svc_pred))

#Feature Importance
rfc.feature_importances_
(pd.Series(rfc.feature_importances_, index=X.columns).plot(kind='barh'))
import pickle

# Firstly we will be using the dump() function to save the model using pickle
saved_model = pickle.dumps(rfc)

# Then we will be loading that saved model
rfc_from_pickle = pickle.loads(saved_model)

# lastly, after loading that model we will use this to make predictions
rfc_from_pickle.predict(X_test)
diabetes_df.head()
diabetes_df.tail()
rfc.predict([[0,137,40,35,168,43.1,2.228,33]]) #4th patient
rfc.predict([[10,101,76,48,180,32.9,0.171,63]])  # 763 th patient