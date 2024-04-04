import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.optimizers import *

# INPUT DATA AS A DATAFRAME FROM A CSV FILE AS I CONVERTED THE DILE FROM EXCEL FILE TO CSV FILE
training = pd.read_csv('C:/Users/ElMoghazy/Desktop/TrackPadData.csv')
train = ['Speeds-Count', 'Block', 'Execution-time']
# print (pd.get_dummies(training['Direction']) )
tmp = pd.get_dummies(training['Direction'])
tmp2 = pd.get_dummies(training['Distance'])
tmp3 = pd.get_dummies(training['LToR-or-RToL'])
tmp4 = pd.get_dummies(training['Speed1'])

X1 = training[train]
list = [X1, tmp, tmp2, tmp3, tmp4]
X = pd.concat(list, axis=1)
y = training['Speed2']
print(y)

# CONVERTIND THE DATAFRAME TO A NUMPY ARRAY
X = X.as_matrix()
y = y.as_matrix().astype(np.float64)
# WILL AUTOMATICALLY SPLIT THE DATA TO TEST AND TRAIN WILL DO THAT LATER
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

"""THE FOLLOWING PARAMETERS MUST BE MODIFIED
ACCORDING TO THE INPUTS OF THE MOUSDATA BUT NEED
FURTHUR INNFORMATION ABOUT THE INPUTS AND THE OOUTPUTS"""

InputDimension = 16
NumOfNeurons1 = 100
NumOfNeurons2 = 100
K.set_image_dim_ordering('th')

model = Sequential()
model.add(Dense(NumOfNeurons1, input_dim=InputDimension, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(NumOfNeurons2, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
optimizer = Adam(lr=0.009, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00003)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
model.fit(X_train, y_train, nb_epoch=500, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=2)
print()
print('Accuracy', scores[1] * 100)