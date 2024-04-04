import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

NumOfOutputs = 2
# INPUT DATA AS A DATAFRAME FROM A CSV FILE AS I CONVERTED THE DILE FROM EXCEL FILE TO CSV FILE
training = pd.read_csv('C:/Users/ElMoghazy/Desktop/MouseData' + str(NumOfOutputs) + '.csv')
train = ['Participant', 'Speed2', 'Speeds-Count', 'Block', 'Execution-time', 'Reg', 'Distance', 'Direction',
         'LToR-or-RToL'         ]
# print (pd.get_dummies(training['Direction']) )
training.loc[training['Distance'] == 'Short', 'Distance'] = 1
training.loc[training['Distance'] == 'Medium', 'Distance'] = 2
training.loc[training['Distance'] == 'Long', 'Distance'] = 3

training.loc[training['Direction'] == 'Horizontal', 'Direction'] = 1
training.loc[training['Direction'] == 'Diagonal', 'Direction'] = 0
training.loc[training['Direction'] == 'Vertical', 'Direction'] = -1

training.loc[training['LToR-or-RToL'] == 'L', 'LToR-or-RToL'] = 1
training.loc[training['LToR-or-RToL'] == 'R', 'LToR-or-RToL'] = -1

X = training[train]
y = training['Speed1']
X = X.astype('float')
# CONVERTIND THE DATAFRAME TO A NUMPY ARRAY
X = X.as_matrix()
y = y.as_matrix()

# WILL AUTOMATICALLY SPLIT THE DATA TO TEST AND TRAIN WILL DO THAT LATER
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf = SVC(C=10, kernel='rbf', degree=10, gamma='auto', coef0=10, shrinking=True, probability=False, tol=0.001,
          cache_size=1000, class_weight='balanced', verbose=0, max_iter=-1, decision_function_shape='ovr',
          random_state=0).fit(X_train,
                              y_train)
print('Accuracy of RBF-kernel SVC on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of RBF-kernel SVC on test set: {:.2f}'.format(clf.score(X_test, y_test)))
