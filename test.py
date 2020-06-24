import numpy as np
from joblib import dump, load

#Load model from joblib file
clf = load('breast-cancer_model.joblib')

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)
