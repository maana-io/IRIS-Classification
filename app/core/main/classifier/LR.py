'''
Logistic Regression
'''
from sklearn.linear_model import LogisticRegression
import numpy as np



class LR(object):

    def __init__(self, penalty=None, dual=None, solver=None, multi_class=None, class_weight=None, fit_intercept=None):
        self.__penalty = penalty
        self.__dual = dual
        self.__solver = solver
        self.__multi_class = multi_class
        self.__class_weight = class_weight
        self.__fit_intercept = fit_intercept

        #required by backward feature selection
        self.coef_ = None
        if penalty is None or dual is None or solver is None or multi_class is None or fit_intercept is None:
            self.__model = None
        else:
            self.__model = LogisticRegression(penalty=self.__penalty, dual=self.__dual, solver=self.__solver, multi_class=self.__multi_class,
                                              class_weight=self.__class_weight, fit_intercept=self.__fit_intercept, verbose=0)



    def fit(self, X, Y):
        self.__model.fit(X, Y)
        #required by backward feature selection
        self.coef_ = self.__model.coef_


    def predict(self, X):
        return self.__model.predict(X)


    def predict_proba(self, X):
        return self.__model.predict_proba(X)


    def get_weights(self, class_no):
        return self.get_all_weights()[class_no]


    def get_intercepts(self):
        if self.__fit_intercept:
            if len(self.__model.classes_) > 2:
                return self.__model.intercept_
            else:
                return np.array([self.__model.intercept_, self.__model.intercept_])
        else:
            return np.array([0.0] * len(self.__model.classes_))

    def get_all_weights(self):
        if len(self.__model.classes_) > 2:
            return self.__model.coef_
        else:
            return np.array([self.__model.coef_[0], self.__model.coef_[0]])


    def get_params(self, deep = True):
        params = {
            'penalty': self.__penalty,
            'dual': self.__dual,
            'solver': self.__solver,
            'multi_class': self.__multi_class,
            'class_weight': self.__class_weight,
            'fit_intercept': self.__fit_intercept,
        }

        #only available after trained
        if hasattr(self.__model, "coef_"):
            params.update({
                'classes': self.__model.classes_.tolist(),
            })
            if self.__fit_intercept:
                if len(self.__model.classes_) > 2:
                    params.update({
                        'intercept': self.__model.intercept_.tolist(),
                    })
                else:
                    params.update({
                        'intercept': [self.__model.intercept_[0], self.__model.intercept_[0]],
                    })
            else:
                params.update({
                    'intercept': [0.0] * len(self.__model.classes_),
                })
            if len(self.__model.classes_) > 2:
                params.update({
                    'coef': self.__model.coef_.tolist(),
                })
            else:
                params.update({
                    'coef': [self.__model.coef_.tolist(), self.__model.coef_.tolist()],
                })
        return params



    def set_params(self, **params):
        if 'classes' in params:
            self.__model.classes_ = np.asarray(params['classes'], dtype=np.int32)
        if 'coef' in params:
            if len(self.__model.classes_) > 2:
                self.__model.coef_ = np.asarray(params['coef'], dtype=np.float64)
            else:
                self.__model.coef_ = np.asarray(params['coef'][0], dtype=np.float64)
        if 'intercept' in params:
            if len(self.__model.classes_) > 2:
                self.__model.intercept_ = np.asarray(params['intercept'], dtype=np.float64)
            else:
                self.__model.intercept_ = np.asarray(params['intercept'][:1], dtype=np.float64)
        if 'penalty' in params:
            self.__penalty = params['penalty']
        if 'dual' in params:
            self.__dual = params['dual']
        if 'solver' in params:
            self.__solver = params['solver']
        if 'multi_class' in params:
            self.__multi_class = params['multi_class']
        if 'class_weight' in params:
            self.__class_weight = params['class_weight']
        if 'fit_intercept' in params:
            self.__fit_intercept = params['fit_intercept']

        return


    def labels(self):
        return self.__model.classes_


    def num_classes(self):
        return len(self.__model.classes_)


    def num_weights(self):
        all_weights = self.get_all_weights()
        return len(all_weights[0])


    def __str__(self):
        return 'Logistic Regression.'
