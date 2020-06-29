'''
Backward stepwise feature selection
'''

from sklearn.feature_selection import RFE



class BackwardStepwise(object):

    def __init__(self, n, estimator, step = 100):
        assert type(n) is int and n > 0, "Invalid parameter type or value %s (number of features)" % n
        assert type(step) is int and step > 0 , "Invalid parameter type or value %s (step)" % n

        self.__estimator = estimator
        self.__n = n
        self.__step = step
        self.__model = RFE(self.__estimator, n_features_to_select = self.__n, step = self.__step)


    def score_features(self, X, Y):
        self.__model.fit(X, Y)
        return self.__model.ranking_


    def select_features(self, X):
        return self.__model.transform(X)


    def __str__(self):
        return '''
        Backward stepwise feature selection:
            Top features selected: %s
            Step size: %s
            Estimator: %s
        ''' % (self.__n, self.__step, self.__estimator)