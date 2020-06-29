'''
SVM
'''
from sklearn.svm import SVC
import numpy as np



class LSVC(object):

    def __init__(self, class_weighting='balanced', C=1.0, max_iter=2):
        self.__class_weight = class_weighting
        self.__C = C
        self.__kernel = 'linear'
        self.__probability = True
        self.__decision_func_shape = 'ovr'
        self.__max_iter=max_iter

        #required by backward feature selection
        self.coef_ = None

        #probability must set to True in order to call predict_proba
        self.__model = SVC(class_weight=self.__class_weight, kernel=self.__kernel, C=self.__C,
                           probability=self.__probability, decision_function_shape = self.__decision_func_shape,
                           max_iter=self.__max_iter, random_state=1078)



    def fit(self, X, Y):
        self.__model.fit(X, Y)
        #required by backward feature selection, only available for linear kernel
        self.coef_ = self.get_all_weights()


    def predict(self, X):
        return self.__model.predict(X)


    def predict_proba(self, X):
        return self.__model.predict_proba(X)


    def get_weights(self, class_no):
        return self.get_all_weights()[class_no]


    #even if OVR is used the intercept_ are still results of OVO
    #return avg OVO intercepts as OVR intercepts - for information only, e.g. intercepts should not be used to make prediction
    def get_intercepts(self):
        num_classes = len(self.__model.classes_)
        if num_classes > 2:
            all_classes_avg_intercept_ = []
            running_seq = 0
            for clsidx in range(num_classes):
                if clsidx == 0:
                    this_class_avg_intercept_ = []
                else:
                    this_class_avg_intercept_ = [intercepts for intercepts in self.__model.intercept_[0:clsidx]]
                for other_clsidx in range(clsidx+1, num_classes):
                    this_class_avg_intercept_.append(self.__model.intercept_[running_seq])
                    running_seq += 1
                this_class_avg_intercept_ = sum(np.array(this_class_avg_intercept_))/(num_classes-1)
                all_classes_avg_intercept_.append(this_class_avg_intercept_)
            return np.array(all_classes_avg_intercept_)
        else:
            return np.array([self.__model.intercept_[0], self.__model.intercept_[0]])


    #even if OVR is used the coef_ are still results of OVO
    #return avg OVO weights as OVR weights - for information only, e.g. weights should not be used to make prediction
    def get_all_weights(self):
        num_classes = len(self.__model.classes_)
        if num_classes > 2:
            all_classes_avg_coef_ = []
            running_seq = 0
            for clsidx in range(num_classes):
                if clsidx == 0:
                    this_class_avg_coef_ = []
                else:
                    this_class_avg_coef_ = [weights for weights in self.__model.coef_[0:clsidx]]
                for other_clsidx in range(clsidx+1, num_classes):
                    this_class_avg_coef_.append(self.__model.coef_[running_seq])
                    running_seq += 1
                this_class_avg_coef_ = sum(np.array(this_class_avg_coef_))/(num_classes-1)
                all_classes_avg_coef_.append(this_class_avg_coef_)
            return np.array(all_classes_avg_coef_)
        else:
            return np.array([self.__model.coef_[0], self.__model.coef_[0]])


    def get_params(self, deep = True):
        params = {
            'C': self.__C,
            'class_weighting': self.__class_weight,
            'max_iter': self.__max_iter,
        }

        #only available after trained
        if hasattr(self.__model, "support_"):
            params.update({
                'support': self.__model.support_.tolist(),
                'support_vectors': self.__model.support_vectors_.tolist(),
                'n_support': self.__model.n_support_.tolist(),
                'dual_coef': self.__model.dual_coef_.tolist(),
                '_dual_coef': self.__model._dual_coef_.tolist(),
                #'coef': self.__model.coef_.tolist(),
                'intercept': self.__model.intercept_.tolist(),
                '_intercept': self.__model._intercept_.tolist(),
                'fit_status': self.__model.fit_status_,
                'classes': self.__model.classes_.tolist(),
                'probA': self.__model.probA_.tolist(),
                'probB': self.__model.probB_.tolist(),
                'class_weight': self.__model.class_weight_.tolist(),
                'shape_fit': self.__model.shape_fit_,
                'sparse': self.__model._sparse,
                'gamma': self.__model._gamma,
                'degree': self.__model.degree,
                'coef0': self.__model.coef0,
                'kernel': self.__model.kernel,
                'impl': self.__model._impl
            })
        return params



    def set_params(self, **params):
        if 'C' in params:
            self.__C = float(params['C'])
        if 'max_iter' in params:
            self.__max_iter = int(params['max_iter'])
        if 'support' in params:
            self.__model.support_  = np.asarray(params['support'], dtype=np.int32)
        if 'support_vectors' in params:
            self.__model.support_vectors_ = np.asarray(params['support_vectors'], dtype=np.float64)
        if 'n_support' in params:
            self.__model.n_support_ = np.asarray(params['n_support'], dtype=np.int32)
        if 'dual_coef' in params:
            self.__model.dual_coef_ = np.asarray(params['dual_coef'], dtype=np.float64)
        if '_dual_coef' in params:
            self.__model._dual_coef_ = np.asarray(params['_dual_coef'], dtype=np.float64)
        # if 'coef' in params:                                          #coef is readonly
        #     self.__model.coef_ = params['coef']
        if 'intercept' in params:
            self.__model.intercept_ = np.asarray(params['intercept'], dtype=np.float64)
        if '_intercept' in params:
            self.__model._intercept_ = np.asarray(params['_intercept'], dtype=np.float64)
        if 'fit_status' in params:
            self.__model.fit_status_ = int(params['fit_status'])
        if 'classes' in params:
            self.__model.classes_ = np.asarray(params['classes'], dtype=np.int32)
        if 'probA' in params:
            self.__model.probA_ = np.asarray(params['probA'], dtype=np.float64)
        if 'probB' in params:
            self.__model.probB_ = np.asarray(params['probB'], dtype=np.float64)
        if 'class_weight' in params:
            self.__model.class_weight_ = np.asarray(params['class_weight'], dtype=np.float64)
        if 'class_weighting' in params:
            self.__class_weight = params['class_weighting']
        if 'shape_fit' in params:
            self.__model.shape_fit_ = np.asarray(params['shape_fit'], dtype=np.int32)
        if 'sparse' in params:
            self.__model._sparse = bool(params['sparse'])
        if 'gamma' in params:
            self.__model._gamma = float(params['gamma'])
        if 'degree' in params:
            self.__model.degree = float(params['degree'])
        if 'coef0' in params:
            self.__model.coef0 = float(params['coef0'])
        if 'kernel' in params:
            self.__model.kernel = params['kernel']
        if 'impl' in params:
            self.__model._impl = params['impl']

        return


    def labels(self):
        return self.__model.classes_


    def num_classes(self):
        return len(self.__model.classes_)


    def num_weights(self):
        all_weights = self.get_all_weights()
        return len(all_weights[0])


    def __str__(self):
        return 'Linear Support Vector Classification.'
