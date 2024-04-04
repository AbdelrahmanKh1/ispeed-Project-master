if __name__ == '__main__':
    import pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC
    import scipy
    import numpy as np

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    if __name__ == '__main__':
        NumOfOutputs = 3
        # INPUT DATA AS A DATAFRAME FROM A CSV FILE AS I CONVERTED THE DILE FROM EXCEL FILE TO CSV FILE
        training = pd.read_csv('C:/Users/ElMoghazy/Desktop/M.csv')
        testing = pd.read_csv('C:/Users/ElMoghazy/Desktop/MouseData' + str(NumOfOutputs) + 'mmmmm.csv')

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
        y = training['Speed1']
        X = X.astype('float')
        # CONVERTIND THE DATAFRAME TO A NUMPY ARRAY
        X_train = X.as_matrix()
        y_train = y.as_matrix()

        X12 = testing[test]
        # print ( testing.corr() )
        list1 = [X12]
        Y = pd.concat(list1, axis=1)
        y1 = testing['Speed1.1']
        Y = Y.astype('float')
        # CONVERTIND THE DATAFRAME TO A NUMPY ARRAY
        X_test = Y.as_matrix()
        y_test = y1.as_matrix()

        # WILL AUTOMATICALLY SPLIT THE DATA TO TEST AND TRAIN WILL DO THAT LATER


        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        c = [2 ** x for x in np.arange(-5, 15, 2).tolist()]

        param_grid = {'C': c}

        # c = np.arange(10e-1, 10e15, 10e2)
        # c.tolist()
        #
        # gamma = np.arange(10e-15, 10e2, 10e2)
        # gamma.tolist()

        # param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10.0, 100.0, 1000.0]
        # param_grid = dict(gamma=param_range, C=param_range)
        # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(
            SVC(C=7, kernel='rbf', degree=10, gamma='auto', coef0=10, shrinking=True, probability=False, tol=0.001,
                cache_size=1000, class_weight='balanced', verbose=10, max_iter=-1, decision_function_shape='ovr',
                random_state=0), param_grid =param_grid, cv=10, n_jobs=-1, scoring='accuracy')
        grid.fit(X_train, y_train)

        print ()
        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))
        print(grid.best_estimator_.score(X_test, y_test))