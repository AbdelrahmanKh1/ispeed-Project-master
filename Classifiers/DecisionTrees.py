import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

NumOfOutputs = 3
# INPUT DATA AS A DATAFRAME FROM A CSV FILE AS I CONVERTED THE DILE FROM EXCEL FILE TO CSV FILE
training = pd.read_csv('C:/Users/ElMoghazy/Desktop/MouseData' + str(NumOfOutputs) + 'Train.csv')
testing = pd.read_csv('C:/Users/ElMoghazy/Desktop/MouseData' + str(NumOfOutputs) + 'Test.csv')


train = ['Participant', 'Speed2', 'Block', 'Execution-time', 'Reg', 'Distance', 'Direction', 'LToR-or-RToL',
         'Relative-Speed']

test = ['Participant', 'Speed2.1', 'Block.1', 'Execution-time.1', 'Reg.1', 'Distance.1', 'Direction.1',
        'LToR-or-RToL.1',
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

X = training[train]
y = training['Speed1']
X = X.astype('float')
# CONVERTIND THE DATAFRAME TO A NUMPY ARRAY
X_train = X.as_matrix()
y_train = y.as_matrix()

X2 = testing[test]
y1 = testing['Speed1.1']
X2 = X2.astype('float')
# CONVERTIND THE DATAFRAME TO A NUMPY ARRAY
X_test = X2.as_matrix()
y_test = y1.as_matrix()



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                             min_impurity_split=1e-07, class_weight='balanced', presort=False).fit(X_train, y_train)

print('Accuracy of RBF-kernel SVC on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of RBF-kernel SVC on test set: {:.2f}'.format(clf.score(X_test, y_test)))
