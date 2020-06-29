'''
Rank features by its correlation with label
   scikit learn methods:
       SelectKBest
       SelectPercentile
       SelectFdr
       SelectFpr
       SelectFwe
       GenericUnivariateSelect
'''

from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2
import numpy as np



class LabelCorrelation(object):

    def __init__(self, n):
        assert type(n) is int and n > 0 or \
               type(n) is float and n > 0 and n <= 1, "Invalid parameter value %s (number of features or percentile)" % n
        self.__n = n
        self.__model = None
        if type(n) is int and n > 0:
            self.__model = GenericUnivariateSelect(chi2, mode='k_best', param = self.__n)
        elif type(n) is float and n > 0 and n <= 1:
            self.__model = GenericUnivariateSelect(chi2, mode='percentile', param = self.__n)



    def score_features(self, X, Y):
        someNegative = np.any(X < 0.0)
        assert not someNegative, "Chisquare correlation requires non-negative feature values."
        self.__model.fit(X, Y)
        return self.__model.scores_



    def select_features(self, X):
        self.__model.transform(X)



    def __str__(self):
        return '''
        Correlation feature selection using chi2 (either k-best or percentile):
            Top features selected: %s 
        ''' % self.__n
