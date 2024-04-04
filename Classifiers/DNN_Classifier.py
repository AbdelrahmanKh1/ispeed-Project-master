import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import callbacks
from keras import optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NumOfOutputs = 2
InputDimension = 8
# INPUT DATA AS A DATAFRAME FROM A CSV FILE AS I CONVERTED THE DILE FROM EXCEL FILE TO CSV FILE

training = pd.read_csv('C:/Users/ElMoghazy/Desktop/MouseData' + str(NumOfOutputs) + 'Train.csv')
testing = pd.read_csv('C:/Users/ElMoghazy/Desktop/MouseData' + str(NumOfOutputs) + 'Test.csv')

train = ['Speed2', 'Block', 'Execution-time', 'Reg', 'Distance', 'Direction', 'LToR-or-RToL', 'Relative-Speed']

test = ['Speed2.1', 'Block.1', 'Execution-time.1', 'Reg.1', 'Distance.1', 'Direction.1', 'LToR-or-RToL.1',
        'Relative-Speed.1']
# print (pd.get_dummies(training['Direction']) )
training.loc[training['Distance'] == 'Short', 'Distance'] = 1
training.loc[training['Distance'] == 'Medium', 'Distance'] = 2
training.loc[training['Distance'] == 'Long', 'Distance'] = 3

training.loc[training['Direction'] == 'Horizontal', 'Direction'] = 1
training.loc[training['Direction'] == 'Diagonal', 'Direction'] = 0
training.loc[training['Direction'] == 'Vertical', 'Direction'] = -1

training.loc[training['LToR-or-RToL'] == 'L', 'LToR-or-RToL'] = 1
training.loc[training['LToR-or-RToL'] == 'R', 'LToR-or-RToL'] = -1

testing.loc[testing['Distance.1'] == 'Short', 'Distance.1'] = 1
testing.loc[testing['Distance.1'] == 'Medium', 'Distance.1'] = 2
testing.loc[testing['Distance.1'] == 'Long', 'Distance.1'] = 3

testing.loc[testing['Direction.1'] == 'Horizontal', 'Direction.1'] = 1
testing.loc[testing['Direction.1'] == 'Diagonal', 'Direction.1'] = 0
testing.loc[testing['Direction.1'] == 'Vertical', 'Direction.1'] = -1

testing.loc[testing['LToR-or-RToL.1'] == 'L', 'LToR-or-RToL.1'] = 1
testing.loc[testing['LToR-or-RToL.1'] == 'R', 'LToR-or-RToL.1'] = -1

X1 = training[train]
list = [X1]
X = pd.concat(list, axis=1)
y = pd.get_dummies((training['Speed1']))
X = X.astype('float')
# CONVERTIND THE DATAFRAME TO A NUMPY ARRAY
X_train = X.as_matrix()
y_train = y.as_matrix()

X12 = testing[test]
# print ( testing.corr() )
list1 = [X12]
Y = pd.concat(list1, axis=1)
y1 = pd.get_dummies(testing['Speed1.1'])
Y = Y.astype('float')
# CONVERTIND THE DATAFRAME TO A NUMPY ARRAY
X_test = Y.as_matrix()
y_test = y1.as_matrix()

"""THE FOLLOWING PARAMETERS MUST BE MODIFIED
ACCORDING TO THE INPUTS OF THE MOUSDATA BUT NEED
FURTHUR INNFORMATION ABOUT THE INPUTS AND THE OOUTPUTS"""

NumOfNeurons1 = 1000
NumOfNeurons2 = 200

model = Sequential()
model.add(Dense(1000, input_dim=InputDimension, activation='relu'))
model.add(Dense(NumOfNeurons2, activation='relu'))
model.add(Dense(NumOfNeurons2, activation='relu'))
model.add(Dense(NumOfNeurons2, activation='relu'))

earlystopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='min')

model.add(Dense(NumOfOutputs, activation='softmax'))

optimizer = optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10000, batch_size=200, validation_split=0.2, verbose=1, callbacks=[earlystopping])
scores = model.evaluate(X_test, y_test)
print()
print('Accuracy', scores[1] * 100)
