import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier

path = "dataset.csv"
df = pd.read_csv(path);


'''
df.info()
print(df.describe())
print(df.head(4))
'''



#one way of finding the important features
'''
array = df.values

X = array[:,0:13]  # input components 
Y = array[:,13]    #output components

from sklearn.ensemble import ExtraTreesClassifier        
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)      
'''




#another way to find important features
import seaborn as sns
corrmat = df.corr()
top_corr_features = corrmat.index 
plt.figure(figsize=(20,20))
g = sns.heatmap(df[top_corr_features].corr(), annot=True)




dataset = pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

standardScaler = StandardScaler()

columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
 





'''Random Forest'''
X = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=6)
 

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train, y_train)

#testing
y_pred = classifier.predict(X_test)

#accuracy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
#print("Confusion Matrix:")
#print(result)
result1 = classification_report(y_test, y_pred)
#print("Classification Report:",)
#print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy of Random Forest:",result2)
 






'''Naive Bayes'''
from sklearn.naive_bayes import GaussianNB 


X = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=6)


gnb = GaussianNB() 
gnb.fit(X_train, y_train) 


#predicting
y_pred = gnb.predict(X_test) 

result2 = accuracy_score(y_test,y_pred)
print("Accuracy of Naive Bayes:",result2)
 





'''Logistic regression'''

X = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=6)

from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train) 

#testing
y_pred = classifier.predict(X_test)

result2 = accuracy_score(y_test,y_pred)
print("Accuracy of Logistic Regression:",result2)


