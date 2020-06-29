'''
Main process for Assisted Categorization
'''
import pandas as pd
import numpy as np
from scipy.special import entr


from app.core.main.classifier.LR import LR
from app.core.main.classifier.LSVC import LSVC
from app.core.main.classifier.Ensemble import Ensemble
from app.core.main.featurizer.Featurizer import Featurizer
from app.core.main.feature_selection.LabelCorrelation import LabelCorrelation
from app.core.main.feature_selection.BackwardStepwise import BackwardStepwise
from app.core.main.evaluator.ModelEvaluator import ModelEvaluator
from sklearn.feature_extraction import stop_words

from app.core.main.tokenizer.BaseTokenizer import BaseTokenizer

defaultModelConfiguration = {
        "type": "LOGISTIC_REGRESSION",              #or LINEAR_SVC
        "class_weight": "balanced",
        "tokenizer": BaseTokenizer(),
        "ngram_range": (1, 1),
        "sublinear_tf": True,
        "smooth_idf": True,
        "penalty": "l2",
        "multi_class": "ovr",
        "solver": "liblinear",
        "dual": True,
        "fit_intercept": True,
        'max_df': 1.,
        'min_df': 0.,
        'stopwords': stop_words.ENGLISH_STOP_WORDS,
        'C': 1.,
        'max_iter': 1000,
        }


class Classifier(object):
    #supported model types
    LR_MODEL_TYPE = "LOGISTIC_REGRESSION"
    SVC_MODEL_TYPE = "LINEAR_SVC"
    ENSEMBLE_SVC_MODEL_TYPE = "ENSEMBLE_LINEAR_SVC"
    ENSEMBLE_LR_MODEL_TYPE = "ENSEMBLE_LOGISTIC_REGRESSION"

    #supported feature selection mode
    CC_fs_correlation = "CORRELATION"
    CC_fs_backward = "RFE"

    supported_feature_selection_modes = [CC_fs_backward, CC_fs_correlation]


    def __init__(self, model_configuration = defaultModelConfiguration):
        self.__model_configuration = model_configuration
        self.__model = None
        self.__featurizer = None
        self.__labeler = None


    def train(self, input_df, schema):
        #debug
        print("train starts ...")

        assert isinstance(input_df, pd.DataFrame)

        field_names, labelFieldName, field_types, train_data, labelData = self.__read_training_data(input_df, schema)

        if self.__model_configuration["type"] == Classifier.LR_MODEL_TYPE:
            m = LR(penalty=self.__model_configuration["penalty"], dual=self.__model_configuration["dual"],
                   solver=self.__model_configuration["solver"], multi_class=self.__model_configuration["multi_class"],
                   class_weight=self.__model_configuration["class_weight"], fit_intercept=self.__model_configuration["fit_intercept"])
        elif self.__model_configuration["type"] == Classifier.SVC_MODEL_TYPE:
            m = LSVC(class_weighting=self.__model_configuration["class_weight"], C=self.__model_configuration["C"],
                     max_iter=self.__model_configuration["max_iter"])
        elif self.__model_configuration["type"] == Classifier.ENSEMBLE_SVC_MODEL_TYPE:
            field_names = list(input_df.columns)
            assert 'TRAIN_GROUP' == field_names[0]
            assert schema[0] in ['NOOP', 'MIN_MAX_SCALER', 'LABEL_ENCODER', 'NUMERICAL', 'BOOLEAN']

            base_model = LSVC(class_weighting=self.__model_configuration["class_weight"], C=self.__model_configuration["C"],
                     max_iter=self.__model_configuration["max_iter"])
            m = Ensemble(base_model, group_index=field_names.index('TRAIN_GROUP'))
        elif self.__model_configuration["type"] == Classifier.ENSEMBLE_LR_MODEL_TYPE:
            field_names = list(input_df.columns)
            assert 'TRAIN_GROUP' == field_names[0]
            assert schema[0] in ['NOOP', 'MIN_MAX_SCALER', 'LABEL_ENCODER', 'NUMERICAL', 'BOOLEAN']

            base_model = LR(penalty=self.__model_configuration["penalty"], dual=self.__model_configuration["dual"],
                            solver=self.__model_configuration["solver"], multi_class=self.__model_configuration["multi_class"],
                            class_weight=self.__model_configuration["class_weight"], fit_intercept=self.__model_configuration["fit_intercept"])
            m = Ensemble(base_model, group_index=field_names.index('TRAIN_GROUP'))
        else:
            m = None

        fm = Featurizer(field_names, field_types,
                        max_df=self.__model_configuration["max_df"],
                        min_df=self.__model_configuration["min_df"],
                        stop_words=self.__model_configuration["stopwords"],
                        sublinear_tf=self.__model_configuration["sublinear_tf"],
                        smooth_idf=self.__model_configuration["smooth_idf"],
                        ngram_range=self.__model_configuration["ngram_range"],
                        tokenizer=self.__model_configuration["tokenizer"]
                        )

        lm = Featurizer([labelFieldName], [Featurizer.FT_Label],
                        max_df=self.__model_configuration["max_df"],
                        min_df=self.__model_configuration["min_df"],
                        stop_words=self.__model_configuration["stopwords"],
                        sublinear_tf=self.__model_configuration["sublinear_tf"],
                        smooth_idf=self.__model_configuration["smooth_idf"],
                        ngram_range=self.__model_configuration["ngram_range"],
                        tokenizer=self.__model_configuration["tokenizer"]
                        )

        features = fm.fit_transform(train_data)
        labels = lm.fit_transform(labelData)

        m.fit(features, labels)
        self.__model = m
        self.__featurizer = fm
        self.__labeler = lm

        #debug
        print("train finished ...")



    def get_models(self):
        return (self.__model, self.__featurizer, self.__labeler)



    def load_models(self, m, fm, lm):
        assert isinstance(m, LR) or  isinstance(m, LSVC) or  isinstance(m, Ensemble), "Expect a LR model or SVM model or Ensemble model."
        assert isinstance(fm, Featurizer), "Expect a Featurizer model."
        assert isinstance(lm, Featurizer), "Expect a Featurizer model."

        self.__model = m
        self.__featurizer = fm
        self.__labeler = lm



    def predict_proba(self, input_df, multilabel_pred = False):
        #debug
        print("predict_proba starts ...")

        assert isinstance(self.__model, LR) or isinstance(self.__model, LSVC) or isinstance(self.__model, Ensemble)
        assert isinstance(self.__featurizer, Featurizer)
        assert isinstance(self.__labeler, Featurizer)

        assert isinstance(input_df, pd.DataFrame)

        field_names = list(input_df.columns)
        assert len(field_names) == len(self.__featurizer.get_schema()), "Input data has different schema than input schema."

        predictors = self.__featurizer.transform(input_df)
        probs = self.__model.predict_proba(predictors)

        if multilabel_pred:
            list_lbl_indices = list(map(self.__multi_labels__, probs))
            assert type(list_lbl_indices) == list
            assert type(list_lbl_indices[0]) == list
            list_labels = list(map(self.__labeler.inverse_transform, list_lbl_indices))
            predicted_labels = list(map(lambda lbls: ','.join(lbls), list_labels))
        else:
            predicted_labels = self.__labeler.inverse_transform(list(map(np.argmax, probs)))

        class_names = self.__labeler.inverse_transform(range(len(probs[0])))
        probs_to_str = list(map(lambda ps: list(map(lambda fval: "%.2f" % fval, ps)), probs))
        probs_with_name = list(map(lambda ps: zip(class_names, ps), probs_to_str))
        cls_probs = list(map(lambda pn: ','.join(list(map(lambda n_p: n_p[0] + ':' + n_p[1], pn))), probs_with_name))

        labels = pd.DataFrame(predicted_labels, columns=["PredictedLabel"])
        probs_df = pd.DataFrame(cls_probs, columns=["Probabilities"])
        entropies_df = pd.DataFrame(list(map(lambda fval: round(fval, 2), entr(probs).sum(axis=1))), columns=["Entropy"])

        #debug
        print("predict_proba finished ...")

        return pd.concat([input_df, labels, probs_df, entropies_df], axis=1)



    def predict_explain(self, input_df, multilabel_pred = False, topN_features = 10):
        assert isinstance(self.__model, LR) or isinstance(self.__model, LSVC) or isinstance(self.__model, Ensemble)
        assert isinstance(self.__featurizer, Featurizer)
        assert isinstance(self.__labeler, Featurizer)

        assert isinstance(input_df, pd.DataFrame)

        field_names = list(input_df.columns)

        # #debug
        # if len(field_names) != len(self.__featurizer.get_schema()):
        #     print('Input data fields:', len(field_names), sorted(field_names))
        #     print('Input schema:', len(self.__featurizer.get_schema()), self.__featurizer.get_schema())
        assert len(field_names) == len(self.__featurizer.get_schema()), "Input data has different schema than input schema."

        predictors = self.__featurizer.transform(input_df)
        probs = self.__model.predict_proba(predictors)
        pred_classes = list(map(np.argmax, probs))

        if multilabel_pred:
            list_lbl_indices = list(map(self.__multi_labels__, probs))
            assert type(list_lbl_indices) == list
            assert type(list_lbl_indices[0]) == list
            list_labels = list(map(self.__labeler.inverse_transform, list_lbl_indices))
            predicted_labels = list(map(lambda lbls: ','.join(lbls), list_labels))
        else:
            predicted_labels = self.__labeler.inverse_transform(pred_classes)

        class_names = self.__labeler.inverse_transform(range(len(probs[0])))
        probs_to_str = list(map(lambda ps: list(map(lambda fval: "%.2f" % fval, ps)), probs))
        probs_with_name = list(map(lambda ps: zip(class_names, ps), probs_to_str))
        cls_probs = list(map(lambda pn: ','.join(list(map(lambda n_p: n_p[0] + ':' + n_p[1], pn))), probs_with_name))

        labels = pd.DataFrame(predicted_labels, columns=["PredictedLabel"])
        probs_df = pd.DataFrame(cls_probs, columns=["Probabilities"])
        entropies_df = pd.DataFrame(list(map(lambda fval: round(fval, 2), entr(probs).sum(axis=1))), columns=["Entropy"])

        weights = self.__model.get_all_weights()
        feat_names = self.__featurizer.get_all_features()

        raw_contributors = list(map(lambda didx_clsno: zip(feat_names, weights[didx_clsno[1]].tolist(),
                               np.asarray(predictors)[didx_clsno[0]]),
                               enumerate(pred_classes)))
        eligible_contributors = list(map(lambda contrib: [(feat, w * inp_s) for (feat, w, inp_s) in contrib if w * inp_s != 0.0 ],
                                    raw_contributors))
        top_contributors = list(map(lambda feats: ';'.join([fname + '=' + str(round(w, 2)) \
                                   for (fname, w) in sorted(feats, key = lambda n_w: abs(n_w[1]), reverse=True)[:topN_features] ]),
                               eligible_contributors))

        contributors = pd.DataFrame(top_contributors, columns=["TopContributors"])

        return pd.concat([input_df, labels, probs_df, entropies_df, contributors], axis=1)



    def learn(self, input_df):
        prob_df = self.predict_proba(input_df)
        sorted_prob_df = prob_df.sort_values(by = ["Entropy"], ascending = False)

        return sorted_prob_df



    def input_qlty(self, input_df, threshold1 = 0.3, threshold2 = 0.5, topN = 10):
        assert isinstance(self.__model, LR) or isinstance(self.__model, LSVC) or isinstance(self.__model, Ensemble)
        assert isinstance(self.__featurizer, Featurizer)
        assert isinstance(self.__labeler, Featurizer)

        assert isinstance(input_df, pd.DataFrame)

        assert threshold1 <= threshold2, "Entropy threshold values are invalid: %f > %f ." % (threshold1, threshold2)
        assert topN > 0, "Number of top contributors %d is invalid." % topN

        weights = self.__model.get_all_weights()
        feat_names = self.__featurizer.get_all_features()

        assert topN <= len(feat_names), "Number of top contributors %d cannot exceed number of features %d" % (topN, len(feat_names))
        field_names = list(input_df.columns)
        feat_field_names = [fld for (fidx, fld) in  enumerate(list(field_names))]

        assert len(feat_field_names) == len(self.__featurizer.get_schema()), "Input data has different schema than input schema."

        X = self.__featurizer.transform(input_df)
        zero_feats = list(map(lambda xrow: xrow == 0.0, np.array(X)))

        probs = self.__model.predict_proba(X)
        top2Indices = np.argsort(probs, axis=1)[np.ix_(range(probs.shape[0]), range(probs.shape[1])[-2:])]
        entropies = entr(probs).sum(axis=1)

        input_qlty_df = pd.DataFrame(list(map(lambda e: "Good" if e <= threshold1 \
            else "OK" if e <= threshold2 else "Bad", entropies)), columns=["InputQlty"])

        #classes features & weights
        feat_weights = list(map(lambda cls1_cls2: zip(feat_names, weights[cls1_cls2[0]], weights[cls1_cls2[1]],
                                                 range(len(feat_names))), top2Indices))
        contributors = list(map(lambda fweights: list(map( lambda feat_coeff1_coeff2_fidx: \
            (feat_coeff1_coeff2_fidx[0], abs(feat_coeff1_coeff2_fidx[1] - feat_coeff1_coeff2_fidx[2]), feat_coeff1_coeff2_fidx[3]),
                                                 fweights)), feat_weights))

        not_existed_contributors = list(map(lambda fweights_zero_fs: [(feat, coeff) for (feat, coeff, fidx) in fweights_zero_fs[0] \
                    if fweights_zero_fs[1][fidx]], zip(contributors, zero_feats)))
        top_contributors = list(map(lambda fweights: sorted(fweights,
                                        key = lambda feat_coeff: -feat_coeff[1])[:topN], not_existed_contributors))

        top_contributors_str = list(map(lambda fweights: ','.join([feat for (feat, w) in fweights]), top_contributors))
        top_contributors_df = pd.DataFrame(top_contributors_str, columns=["SuggestedFeatures"])

        return pd.concat([input_df, input_qlty_df, top_contributors_df], axis=1)





    def eval(self, input_df, schema, mode, nfolds, topN = 1):

        field_names, label_field_name, field_types, train_data, label_data = self.__read_training_data(input_df, schema)

        if self.__model_configuration["type"] == Classifier.LR_MODEL_TYPE:
            m = LR(penalty=self.__model_configuration["penalty"], dual=self.__model_configuration["dual"],
                   solver=self.__model_configuration["solver"], multi_class=self.__model_configuration["multi_class"],
                   class_weight=self.__model_configuration["class_weight"], fit_intercept=self.__model_configuration["fit_intercept"])
        elif self.__model_configuration["type"] == Classifier.SVC_MODEL_TYPE:
            m = LSVC(class_weighting=self.__model_configuration["class_weight"], C=self.__model_configuration["C"],
                     max_iter=self.__model_configuration["max_iter"])
        elif self.__model_configuration["type"] == Classifier.ENSEMBLE_SVC_MODEL_TYPE:
            field_names = list(input_df.columns)
            assert 'TRAIN_GROUP' == field_names[0]
            assert schema[0] in ['NOOP', 'MIN_MAX_SCALER', 'LABEL_ENCODER', 'NUMERICAL', 'BOOLEAN']

            base_model = LSVC(class_weighting=self.__model_configuration["class_weight"], C=self.__model_configuration["C"],
                              max_iter=self.__model_configuration["max_iter"])
            m = Ensemble(base_model, group_index=field_names.index('TRAIN_GROUP'))
        elif self.__model_configuration["type"] == Classifier.ENSEMBLE_LR_MODEL_TYPE:
            field_names = list(input_df.columns)
            assert 'TRAIN_GROUP' == field_names[0]
            assert schema[0] in ['NOOP', 'MIN_MAX_SCALER', 'LABEL_ENCODER', 'NUMERICAL', 'BOOLEAN']

            base_model = LR(penalty=self.__model_configuration["penalty"], dual=self.__model_configuration["dual"],
                            solver=self.__model_configuration["solver"], multi_class=self.__model_configuration["multi_class"],
                            class_weight=self.__model_configuration["class_weight"], fit_intercept=self.__model_configuration["fit_intercept"])
            m = Ensemble(base_model, group_index=field_names.index('TRAIN_GROUP'))
        else:
            m = None

        fm = Featurizer(field_names, field_types,
                        max_df=self.__model_configuration["max_df"],
                        min_df=self.__model_configuration["min_df"],
                        stop_words=self.__model_configuration["stopwords"],
                        sublinear_tf=self.__model_configuration["sublinear_tf"],
                        smooth_idf=self.__model_configuration["smooth_idf"],
                        ngram_range=self.__model_configuration["ngram_range"],
                        tokenizer=self.__model_configuration["tokenizer"]
                        )

        lm = Featurizer([label_field_name], [Featurizer.FT_Label],
                        max_df=self.__model_configuration["max_df"],
                        min_df=self.__model_configuration["min_df"],
                        stop_words=self.__model_configuration["stopwords"],
                        sublinear_tf=self.__model_configuration["sublinear_tf"],
                        smooth_idf=self.__model_configuration["smooth_idf"],
                        ngram_range=self.__model_configuration["ngram_range"],
                        tokenizer=self.__model_configuration["tokenizer"]
                        )

        evaluator = ModelEvaluator(m, fm, lm, train_data, label_data, topN)
        eval_res = evaluator.eval(mode, nfolds)

        return eval_res




    def eval_data(self, input_df, schema, mode, nfolds, topN = 1):

        field_names, label_field_name, field_types, train_data, label_data = self.__read_training_data(input_df, schema)

        if self.__model_configuration["type"] == Classifier.LR_MODEL_TYPE:
            m = LR(penalty=self.__model_configuration["penalty"], dual=self.__model_configuration["dual"],
                   solver=self.__model_configuration["solver"], multi_class=self.__model_configuration["multi_class"],
                   class_weight=self.__model_configuration["class_weight"], fit_intercept=self.__model_configuration["fit_intercept"])
        elif self.__model_configuration["type"] == Classifier.SVC_MODEL_TYPE:
            m = LSVC(class_weighting=self.__model_configuration["class_weight"], C=self.__model_configuration["C"],
                     max_iter=self.__model_configuration["max_iter"])
        elif self.__model_configuration["type"] == Classifier.ENSEMBLE_SVC_MODEL_TYPE:
            field_names = list(input_df.columns)
            assert 'TRAIN_GROUP' == field_names[0]
            assert schema[0] in ['NOOP', 'MIN_MAX_SCALER', 'LABEL_ENCODER', 'NUMERICAL', 'BOOLEAN']

            base_model = LSVC(class_weighting=self.__model_configuration["class_weight"], C=self.__model_configuration["C"],
                              max_iter=self.__model_configuration["max_iter"])
            m = Ensemble(base_model, group_index=field_names.index('TRAIN_GROUP'))
        elif self.__model_configuration["type"] == Classifier.ENSEMBLE_LR_MODEL_TYPE:
            field_names = list(input_df.columns)
            assert 'TRAIN_GROUP' == field_names[0]
            assert schema[0] in ['NOOP', 'MIN_MAX_SCALER', 'LABEL_ENCODER', 'NUMERICAL', 'BOOLEAN']

            base_model = LR(penalty=self.__model_configuration["penalty"], dual=self.__model_configuration["dual"],
                            solver=self.__model_configuration["solver"], multi_class=self.__model_configuration["multi_class"],
                            class_weight=self.__model_configuration["class_weight"], fit_intercept=self.__model_configuration["fit_intercept"])
            m = Ensemble(base_model, group_index=field_names.index('TRAIN_GROUP'))
        else:
            m = None

        fm = Featurizer(field_names, field_types,
                        max_df=self.__model_configuration["max_df"],
                        min_df=self.__model_configuration["min_df"],
                        stop_words=self.__model_configuration["stopwords"],
                        sublinear_tf=self.__model_configuration["sublinear_tf"],
                        smooth_idf=self.__model_configuration["smooth_idf"],
                        ngram_range=self.__model_configuration["ngram_range"],
                        tokenizer=self.__model_configuration["tokenizer"]
                        )

        lm = Featurizer([label_field_name], [Featurizer.FT_Label],
                        max_df=self.__model_configuration["max_df"],
                        min_df=self.__model_configuration["min_df"],
                        stop_words=self.__model_configuration["stopwords"],
                        sublinear_tf=self.__model_configuration["sublinear_tf"],
                        smooth_idf=self.__model_configuration["smooth_idf"],
                        ngram_range=self.__model_configuration["ngram_range"],
                        tokenizer=self.__model_configuration["tokenizer"]
                        )

        evaluator = ModelEvaluator(m, fm, lm, train_data, label_data, topN)
        return evaluator.eval_data(mode, nfolds, output_dict=True)





    def feature_ranking(self, input_df, schema, mode):
        field_names, labelFieldName, field_types, train_data, labelData = self.__read_training_data(input_df, schema)

        assert mode in Classifier.supported_feature_selection_modes, \
               "Invalid feature selection mode %s. Supported modes are %s" % \
                   (mode, Classifier.supported_feature_selection_modes)

        fm = Featurizer(field_names, field_types,
                        max_df=self.__model_configuration["max_df"],
                        min_df=self.__model_configuration["min_df"],
                        stop_words=self.__model_configuration["stopwords"],
                        sublinear_tf=self.__model_configuration["sublinear_tf"],
                        smooth_idf=self.__model_configuration["smooth_idf"],
                        ngram_range=self.__model_configuration["ngram_range"],
                        tokenizer=self.__model_configuration["tokenizer"]
                        )

        lm = Featurizer([labelFieldName], [Featurizer.FT_Label],
                        max_df=self.__model_configuration["max_df"],
                        min_df=self.__model_configuration["min_df"],
                        stop_words=self.__model_configuration["stopwords"],
                        sublinear_tf=self.__model_configuration["sublinear_tf"],
                        smooth_idf=self.__model_configuration["smooth_idf"],
                        ngram_range=self.__model_configuration["ngram_range"],
                        tokenizer=self.__model_configuration["tokenizer"]
                        )

        X = fm.fit_transform(train_data)
        Y = lm.fit_transform(labelData)

        if self.__model_configuration["type"] == Classifier.LR_MODEL_TYPE:
            m = LR(penalty=self.__model_configuration["penalty"], dual=self.__model_configuration["dual"],
                   solver=self.__model_configuration["solver"], multi_class=self.__model_configuration["multi_class"],
                   class_weight=self.__model_configuration["class_weight"], fit_intercept=self.__model_configuration["fit_intercept"])
        elif self.__model_configuration["type"] == Classifier.SVC_MODEL_TYPE:
            m = LSVC(class_weighting=self.__model_configuration["class_weight"], C=self.__model_configuration["C"],
                     max_iter=self.__model_configuration["max_iter"])
        elif self.__model_configuration["type"] == Classifier.ENSEMBLE_SVC_MODEL_TYPE:
            field_names = list(input_df.columns)
            assert 'TRAIN_GROUP' == field_names[0]
            assert schema[0] in ['NOOP', 'MIN_MAX_SCALER', 'LABEL_ENCODER', 'NUMERICAL', 'BOOLEAN']

            base_model = LSVC(class_weighting=self.__model_configuration["class_weight"], C=self.__model_configuration["C"],
                              max_iter=self.__model_configuration["max_iter"])
            m = Ensemble(base_model, group_index=field_names.index('TRAIN_GROUP'))
        elif self.__model_configuration["type"] == Classifier.ENSEMBLE_LR_MODEL_TYPE:
            field_names = list(input_df.columns)
            assert 'TRAIN_GROUP' == field_names[0]
            assert schema[0] in ['NOOP', 'MIN_MAX_SCALER', 'LABEL_ENCODER', 'NUMERICAL', 'BOOLEAN']

            base_model = LR(penalty=self.__model_configuration["penalty"], dual=self.__model_configuration["dual"],
                            solver=self.__model_configuration["solver"], multi_class=self.__model_configuration["multi_class"],
                            class_weight=self.__model_configuration["class_weight"], fit_intercept=self.__model_configuration["fit_intercept"])
            m = Ensemble(base_model, group_index=field_names.index('TRAIN_GROUP'))
        else:
            m = None

        fs = LabelCorrelation(n = 1.0) if mode == Classifier.CC_fs_correlation \
            else BackwardStepwise(n = 1, step = 1, estimator=m) #(n = 1, step = max(X.shape[1]/300, 1))

        feature_scores = fs.score_features(X, Y)
        feature_scores_rounded = list(map(lambda val: round(val, 2), feature_scores))
        feature_names = fm.get_all_features()

        features_df = pd.DataFrame(zip(feature_names, feature_scores_rounded),
                                   columns=["Feature", "Score"])
        sorted_features_df = features_df.sort_values(by = ["Score"], ascending = False)

        return sorted_features_df




    def model_visualization(self, topN = 10):
        assert topN > 0, "Number of top contributors %d is invalid." % topN
        assert isinstance(self.__model, LR) or isinstance(self.__model, LSVC) or isinstance(self.__model, Ensemble)
        assert isinstance(self.__featurizer, Featurizer)
        assert isinstance(self.__labeler, Featurizer)

        feat_names = self.__featurizer.get_all_features()

        assert topN <= len(feat_names), "Number of top contributors %d cannot exceed number of features %d" \
                                        % (topN, len(feat_names))

        weights = self.__model.get_all_weights()
        feature_weights = list(map(lambda ws: zip(feat_names, ws), weights))
        top_weights = list(map(lambda class_weights: sorted(class_weights, key = lambda fname_w: -abs(fname_w[1]))[:topN], feature_weights))

        num_classes = self.num_classes()
        labels = self.__labeler.inverse_transform(range(num_classes))
        flatten_lbl_weights = [(lbl, fname, w) for (lbl, f_weights) in zip(labels, top_weights)
                                              for (fname, w) in f_weights]
        res_df = pd.DataFrame(flatten_lbl_weights, columns=["Class", "Feature", "Weight"])

        return res_df



    def labels(self):
        return self.__labeler.inverse_transform(self.__model.labels())


    def num_classes(self):
        return self.__model.num_classes()


    def __read_training_data(self, input_df, schema):
        assert isinstance(input_df, pd.DataFrame)

        field_names = list(input_df.columns)

        assert len(field_names) == len(schema), "Input data has different schema than input schema, e.g. length of %d vs %d" \
            % (len(field_names), len(schema))

        #ignore data frame field types, and make a copy
        field_types = [fld for fld in schema]

        assert len([type for type in field_types if type == Featurizer.FT_Label]) == 1, \
            "There must be exactly one field with %s type in training data.(%s)" % (Featurizer.FT_Label, field_types)

        label_field_no = field_types.index(Featurizer.FT_Label)
        label_field_name = field_names[label_field_no]
        label_data = input_df.filter(items=[input_df.columns[label_field_no]])


        del input_df[input_df.columns[label_field_no]]
        field_names.remove(field_names[label_field_no])
        field_types.remove(field_types[label_field_no])

        return (field_names, label_field_name, field_types, input_df, label_data)



    def __multi_labels__(self, probabilities):
        prob_with_index = zip(probabilities, range(len(probabilities)))
        sorted_prob_with_index = sorted(prob_with_index, key = lambda prob_idx: -prob_idx[0])
        err = np.zeros(len(probabilities), dtype=np.float)
        for split_idx in range(len(probabilities)):
            left_ = np.array([p for (p, _) in sorted_prob_with_index[:split_idx]])
            right_ = np.array([p for (p, _) in sorted_prob_with_index[split_idx:]])
            err1 = 0 if len(left_) < 2 else sum(abs(left_ - np.average(left_)))
            err2 = 0 if len(right_) < 2 else sum(abs(right_ - np.average(right_)))
            err[split_idx] = err1 + err2
        opt_split_idx = max(np.argmin(err), 1)

        res = [cls_idx for (_, cls_idx) in sorted_prob_with_index[:opt_split_idx]]
        assert len(res) > 0
        return res



    def __str__(self):
        return 'Classification Service.'




