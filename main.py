'''This is a classification neural network using SciKit Learning Modules'''

#modules
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score, recall_score, precision_score, f1_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

%matplotlib inline

#Import data
from google.colab import drive
drive.mount('/content/drive')

#Turn data to pandas
df = pd.read_csv('/content/drive/My Drive/HorseRacing/Final.csv')
df.shape
df.head()

#Clean data and Encode
df = df.drop(['Comments'] , axis = 1) # removes comments
df = df.apply(preprocessing.LabelEncoder().fit_transform)
X = df.drop(['FPos'], axis=1)

y = df.FPos

#Test, Train Split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=0)

#Trainging The Model
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

#Work out Predictions
print(confusion_matrix(y_train,predict_train))

print(classification_report(y_train,predict_train))

print(confusion_matrix(y_test,predict_test))

print(classification_report(y_test,predict_test))