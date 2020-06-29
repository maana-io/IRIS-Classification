'''
Ensemble model
'''
import numpy as np
import copy


class Ensemble(object):

    def __init__(self, base_model, group_index):
        self.__base_model = base_model
        self.__group_index = group_index
        self.__models = []
        self.__labels = []
        self.__num_features = 0

        #required by backward feature selection
        self.coef_ = None



    def fit(self, _X, _Y):
        X = np.asarray(_X)
        Y = np.asarray(_Y)

        self.__num_features = len(X[0]) - 1                         #excl group
        assert self.__group_index < self.__num_features + 1

        self.__labels = sorted(set(Y))

        featureIndices = list(range(self.__num_features + 1))
        featureIdxWoGrp = featureIndices[:self.__group_index] + featureIndices[self.__group_index+1:]

        self.__models = []
        groups = set(X[:, self.__group_index].tolist())
        assert len(groups) > 1

        for grp in groups:
            Xp = X[X[:, self.__group_index] != grp]
            Xpp = Xp[:, featureIdxWoGrp]
            Yp = Y[X[:, self.__group_index] != grp]

            numUniqueLabels = len(set(list(Yp)))
            if numUniqueLabels > 1:
                print('Train a model with ', Xpp.shape, ' data and ', len(Yp), ' labels.')
                m = copy.deepcopy(self.__base_model)
                m.fit(Xpp, Yp)
                self.__models.append(m)
            else:
                print('Skip a model of single label: ', set(list(Yp)), ', e.g ignoring ', len(Yp), ' data points.')

        print('Total ', len(self.__models), ' models trained and saved.')

        #required by backward feature selection
        self.coef_ = self.get_all_weights()



    def predict(self, _X):
        X = np.asarray(_X)

        featureIndices = list(range(self.__num_features + 1))
        featureIdxWoGrp = featureIndices[:self.__group_index] + featureIndices[self.__group_index+1:]
        Xp = X[:, featureIdxWoGrp]

        all_predictions = []
        assert(len(self.__models) > 0)

        for m in self.__models:
            all_predictions.append(m.predict(Xp))
        all_predictions = np.array(all_predictions).transpose()

        #count votes by rows, tie-break arbitrarily
        return np.apply_along_axis(lambda votes: np.argmax(np.bincount(votes)), 1, all_predictions)


    #
    #
    # Probability = number of votes (of sub-models)/ number of sub-models.
    #               (a vote of 51% has same contribution as a vote of 99%)
    #
    #
    # def predict_proba(self, _X):
    #     X = np.asarray(_X)
    #
    #     featureIndices = list(range(self.__num_features + 1))
    #     featureIdxWoGrp = featureIndices[:self.__group_index] + featureIndices[self.__group_index+1:]
    #     Xp = X[:, featureIdxWoGrp]
    #
    #     all_predictions = []
    #     assert(len(self.__models) > 0)
    #
    #     for m in self.__models:
    #         all_predictions.append(m.predict(Xp))
    #     all_predictions = np.array(all_predictions).transpose()
    #
    #     num_classes = self.num_classes()
    #     probs = np.apply_along_axis(lambda votes: np.bincount(votes, minlength=num_classes)/sum(np.bincount(votes)),
    #                                 1, all_predictions)
    #
    #     return probs


    #
    #
    #  Returns average probabilities of sub-models
    #
    #
    def predict_proba(self, _X):
        X = np.asarray(_X)

        featureIndices = list(range(self.__num_features + 1))
        featureIdxWoGrp = featureIndices[:self.__group_index] + featureIndices[self.__group_index+1:]
        Xp = X[:, featureIdxWoGrp]
        all_labels = self.labels()

        all_predictions = []
        assert(len(self.__models) > 0)

        for m in self.__models:
            probs = m.predict_proba(Xp)
            full_probs = np.zeros((len(Xp), len(all_labels)))
            m_labels = m.labels()
            for clsidx, clsprob in enumerate(probs.transpose()):
                full_probs[:, all_labels.index(m_labels[clsidx])] = clsprob
            all_predictions.append(full_probs)

        preds = np.array(sum(all_predictions)/len(all_predictions))

        return preds



    def get_weights(self, class_no):
        return self.get_all_weights()[class_no]



    def get_intercepts(self):
        assert len(self.__models) > 0
        num_classes = self.num_classes()

        all_models_intercepts = []
        for m in self.__models:
            m_labels = m.labels()
            m_intercepts = np.asarray(m.get_intercepts())
            full_intercepts = np.zeros((num_classes, ))

            for idx, intercept in enumerate(m_intercepts):
                full_intercepts[self.__labels.index(m_labels[idx])] = intercept

            all_models_intercepts.append(full_intercepts)

        assert len(all_models_intercepts) > 0
        return sum(np.array(all_models_intercepts))/len(all_models_intercepts)


    #OVR should be used
    def get_all_weights(self):
        assert len(self.__models) > 0
        num_classes = self.num_classes()
        all_models_weights = []
        for m in self.__models:
            m_labels = m.labels()
            m_weights = np.asarray(m.get_all_weights())
            full_weights = np.zeros((num_classes, self.__num_features))
            for idx, weights in enumerate(m_weights):
                full_weights[self.__labels.index(m_labels[idx])] = weights

            all_models_weights.append(full_weights)

        assert len(all_models_weights) > 0
        return sum(np.array(all_models_weights))/len(all_models_weights)



    def get_params(self, deep = True):
        params = {
            'base_model': self.__base_model,
            'group_index': int(self.__group_index),
        }

        #available only after trained
        if len(self.__models) > 0:
            all_models_params = []
            for m in self.__models:
                all_models_params.append(m.get_params())
            params.update({'all_models_params': all_models_params})
            params.update({'labels': [int(lbl) for lbl in self.__labels]})
            params.update({'num_features': int(self.__num_features)})
            params.update({'coef_': self.coef_.tolist()})
            params.update({'base_model': self.__base_model.get_params()})

        return params



    def set_params(self, **params):
        if 'group_index' in params:
            self.__group_index = int(params['group_index'])
        self.__models = []
        for m_params in params['all_models_params']:
            m = copy.deepcopy(self.__base_model)
            m.set_params(**m_params)
            self.__models.append(m)
        self.__labels = params['labels']
        self.__num_features = int(params['num_features'])
        self.coef_ =  np.asarray(params['coef_'], dtype=np.float64)
        return


    def labels(self):
        return self.__labels


    def num_classes(self):
        return len(self.labels())


    def num_weights(self):
        all_weights = self.get_all_weights()
        return len(all_weights[0])


    def __str__(self):
        return 'Ensemble Classification based on ' + str(self.__base_model)
