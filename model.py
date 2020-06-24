import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from joblib import dump, load

df = pd.read_csv('breast-cancer-wisconsin.csv')

# Preprocess the data
df.replace('?',-99999, inplace=True)
print(df.axes)

df.drop(['id'], 1, inplace=True)
df.drop(['Unnamed: 0'], 1, inplace=True)
# Create X and Y datasets for training
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


clf = RandomForestClassifier()

clf.fit(X_train, y_train)
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)
print("Train accuracy : " + str(train_accuracy))
print("Test accuracy : " + str(test_accuracy))

#Save model as a joblib file
dump(clf, 'breast-cancer_model.joblib')