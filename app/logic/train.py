
import json
import numpy as np
#import pprint

from app.core.main.tokenizer.BaseTokenizer import BaseTokenizer
from app.core.main.tokenizer.PorterTokenizer import PorterTokenizer
from app.core.main.tokenizer.LemmaTokenizer import LemmaTokenizer
from app.core.main.featurizer.Doc2Vector import Doc2Vector
from app.core.main.Classifier import Classifier

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import *


from app.core.main.classifier.LR import LR
from app.core.main.classifier.LSVC import LSVC
from app.core.main.classifier.Ensemble import Ensemble

from app.core.main.featurizer.Featurizer import Featurizer


from app.logic.helpers import *




def evaluate(training_task):
    labeled_data = training_task['data']
    candidate = training_task['candidate']
    model_sel = training_task['modelSelectionParams']

    input_df = datasetToDataframe(labeled_data)

    features = candidate["features"]
    featurizers = candidate["featurizers"]
    config = candidate["config"]

    ac = create_classifier(config)

    filtered_df = input_df.filter([f['name'] for f in features])

    labels, _, _, conf_mat, cls_report = ac.eval_data(input_df=filtered_df, schema=featurizers,
                                                      mode = model_sel['evalMode'], nfolds=model_sel['numFolds'])

    class_performances = []
    for (lbl_idx, lbl) in enumerate(labels):
        cls_buckets = []
        conf_mat_data = conf_mat[lbl_idx]
        for (idx, nval) in enumerate(conf_mat_data):
            cls_buckets.append({'id': id(), 'trueLabel': lbl, 'predictedLabel': labels[idx], 'numInstances': int(nval), 'weight': 1 })

        perf = {
                'id': id(),
                'label': lbl,
                'weight': 1,
                'numInstances': int(cls_report[lbl]['support']),
                'classifiedAs': cls_buckets,
                'recall': float(cls_report[lbl]['recall']),
                'precision': float(cls_report[lbl]['precision']),
                'f1': float(cls_report[lbl]['f1-score']),
                }
        class_performances.append(perf)

    #type ModelPerformance
    pres =  {
        'id': id(),
        'classPerformances': class_performances,
        'numInstances': int(cls_report['weighted avg']['support']),
        'avgRecall': float(cls_report['weighted avg']['recall']),
        'avgPrecision': float(cls_report['weighted avg']['precision']),
        'avgF1': float(cls_report['weighted avg']['f1-score']),
    }

    return pres




def train(training_task):
    #debug
    print("train (train.py) starts ...")

    labeled_data = training_task['data']
    candidate = training_task['candidate']

    input_df = datasetToDataframe(labeled_data)

    features = candidate["features"]
    featurizers = candidate["featurizers"]
    config = candidate["config"]

    filtered_df = input_df.filter([f['name'] for f in features])

    ac = create_classifier(config)

    ac.train(input_df=filtered_df, schema=featurizers)
    mainModel, featurizer, labelEncoder = ac.get_models()

    featurizer_models, _, featurizer_end_offset, _ = featurizer.get_params()

    weights = mainModel.get_all_weights()
    num_weights = mainModel.num_weights()

    labels = list(ac.labels())
    intercepts = mainModel.get_intercepts()

    assert (labels is not None)
    assert (labels[0] is not None)

    class_weights = []
    for cls_idx, lbl in enumerate(labels):
        feature_weights = []
        for fidx in range(len(features)-1):
            feat = features[fidx]
            _start_widx = featurizer_end_offset[fidx - 1] if fidx > 0 else 0
            _end_widx = featurizer_end_offset[fidx]
            feature_weights.append({'id': id(), 'feature': feat, 'weights': list(weights[cls_idx][_start_widx: _end_widx]) })
        class_weights.append({'id': id(), 'class': lbl, 'weights': feature_weights, 'intercept': float(intercepts[cls_idx])})

    featurizers = []
    for fidx, fmodel in enumerate(featurizer_models):
        if fmodel is None:
            featurizers.append(
                {
                    'id': id(),
                    'noop': { 'id': id() }
                }
            )
        elif isinstance(fmodel, MinMaxScaler):
            featurizers.append(
                {
                    'id': id(),
                    'min_max_scaler': {
                        'id': id(),
                        'minValue': float(fmodel.min_[0]),
                        'maxValue': None,
                        'scale': float(fmodel.scale_[0]),
                        'dataMin': float(fmodel.data_min_[0]),
                        'dataMax': float(fmodel.data_max_[0])
                    }
                }
            )
        elif isinstance(fmodel, LabelBinarizer):
            featurizers.append(
                {
                    'id': id(),
                    'label_binarizer': { 'id': id(),
                                         'labels': list(fmodel.classes_)  }
                }
            )
        elif isinstance(fmodel, TfidfVectorizer):
            term_feature_map = []
            for (term, fidx) in fmodel.vocabulary_.items():
                term_feature_map.append({'id': id(), 'term': term, 'featureIdx': int(fidx)})
            featurizers.append(
                {
                    'id': id(),
                    'tfidf_vectorizer': {
                        'id': id(),
                        'vocab': term_feature_map,
                        'idf': [float(idf_val) for idf_val in fmodel.idf_],
                        'stopwords': list(fmodel.stop_words_)
                    }
                }
            )
        elif isinstance(fmodel, MultiLabelBinarizer):
            featurizers.append(
                {
                    'id': id(),
                    'multilabel_binarizer': { 'id': id(),
                                              'labels': list(fmodel.classes_)  }
                }
            )
        elif isinstance(fmodel, LabelEncoder):
            assert (fmodel.classes_ is not None)
            assert (fmodel.classes_[0] is not None)
            featurizers.append(
                {
                    'id': id(),
                    'label_encoder': { 'id': id(),
                                       'labels': list(fmodel.classes_)  }
                }
            )
        elif isinstance(fmodel, Doc2Vector):
            featurizers.append(
                {
                    'id': id(),
                    'doc_to_vector': {
                        'id': id(),
                        'modelFile': fmodel.model_file_,
                        'maxNumWords': int(fmodel.max_num_words_)
                    }
                }
            )

    perf = evaluate(training_task)

    model_data = mainModel.get_params()

    rmodl = {
        'id': id(),
        'type': config["type"],
        'candidate': candidate,
        'labels': labels,
        'learnedWeights': class_weights,
        'learnedFeaturizers': featurizers,
        'labelEncoder':{ 'id': id(), 'labels': labels  },
        'degreeOfFreedom': num_weights,
        'performance': perf,
        'json': json.dumps(model_data)
    }

    #debug
    print("train (train.py) finished ...")

    #type Model
    return rmodl




def loadTrainedModel(model):
    #debug
    print("loadTrainedModel starts ...")

    config = model["candidate"]["config"]
    featurizers = model["learnedFeaturizers"]
    labelEncoder = model["labelEncoder"]
    learnedWeights = model["learnedWeights"]
    labels = model["labels"]
    modelType = model["type"]

    if modelType in [Classifier.LR_MODEL_TYPE, Classifier.SVC_MODEL_TYPE, Classifier.ENSEMBLE_LR_MODEL_TYPE, Classifier.ENSEMBLE_SVC_MODEL_TYPE]:
        #extract labels, coefs, intercepts, feature names and types from learned weights
        num_classes = len(labels)
        num_features = len(learnedWeights[0]["weights"])

        if num_classes > 2:
            num_submodels = num_classes
        else:
            num_submodels = 1

        featureNames = []
        for clsidx in range(num_submodels):
            for featidx in range(num_features):
                if clsidx == 0:
                    featureNames.append(learnedWeights[clsidx]["weights"][featidx]["feature"]["name"])

        if modelType == Classifier.LR_MODEL_TYPE:
            #initialize LR model
            clsModel = LR(penalty=config["penalty"].lower(), dual=config["primal_dual"]=="DUAL", solver=config["solver"].lower(),
                    multi_class=config["multiclass"].lower(), class_weight=config["weighting"].lower(),
                    fit_intercept=config["fitIntercept"])
            #clsModel.set_params(classes=labels, coef=coefficients, intercept=intercepts)
            model_data = json.loads(model['json'])
            clsModel.set_params(**model_data)
        elif modelType == Classifier.SVC_MODEL_TYPE:
            clsModel = LSVC(class_weighting=config["weighting"].lower(), C=config["C"], max_iter=config["max_iter"])
            model_data = json.loads(model['json'])
            clsModel.set_params(**model_data)
        elif modelType == Classifier.ENSEMBLE_LR_MODEL_TYPE:
            baseModel = LR(penalty=config["penalty"].lower(), dual=config["primal_dual"]=="DUAL", solver=config["solver"].lower(),
                          multi_class=config["multiclass"].lower(), class_weight=config["weighting"].lower(),
                          fit_intercept=config["fitIntercept"])
            model_data = json.loads(model['json'])
            clsModel = Ensemble(baseModel, model_data['group_index'])
            clsModel.set_params(**model_data)
        elif modelType == Classifier.ENSEMBLE_SVC_MODEL_TYPE:
            baseModel = LSVC(class_weighting=config["weighting"].lower(), C=config["C"], max_iter=config["max_iter"])
            model_data = json.loads(model['json'])
            clsModel = Ensemble(baseModel, model_data['group_index'])
            clsModel.set_params(**model_data)

        stop_words = ENGLISH_STOP_WORDS if config["stopwords"] == "ENGLISH" else []

        tokenizer = BaseTokenizer() if config["tokenizer"] == "WORD_TOKENIZER" \
             else PorterTokenizer() if config["tokenizer"] == "STEMMER" \
             else LemmaTokenizer()  if config["tokenizer"] == "LEMMATIZER" \
             else None

        ngram_range = (1, 1) if config["ngrams"] == "UNIGRAM" \
                 else (2, 2) if config["ngrams"] == "BIGRAM" \
                 else (1, 2) if config["ngrams"] == "BOTH" \
                 else None

        #initialize featurizers
        featurizer_models = []
        featurizer_data = []
        featurizer_offsets = []
        feat_offset = 0
        featureTypes = []
        for f_rizer in featurizers:
            if 'noop' in f_rizer and f_rizer["noop"] is not None:
                m = None
                feat_offset += 1
                featurizer_data.append(None)
                featureTypes.append("NOOP")
            elif 'min_max_scaler' in f_rizer and f_rizer["min_max_scaler"] is not None:
                m = MinMaxScaler()
                m.min_ = np.ndarray((1,), dtype=np.float)
                m.min_[0] = f_rizer["min_max_scaler"]["minValue"]
                m.scale_ = np.ndarray((1,), dtype=np.float)
                m.scale_[0] = f_rizer["min_max_scaler"]["scale"]
                m.data_min_ = np.ndarray((1,), dtype=np.float)
                m.data_min_[0] = f_rizer["min_max_scaler"]["dataMin"]
                m.data_max_ = np.ndarray((1,), dtype=np.float)
                m.data_max_[0] = f_rizer["min_max_scaler"]["dataMax"]
                feat_offset += 1
                featurizer_data.append(None)
                featureTypes.append("MIN_MAX_SCALER")
            elif 'label_binarizer' in f_rizer and f_rizer["label_binarizer"] is not None:
                m = LabelBinarizer()
                m.classes_ = np.array(f_rizer["label_binarizer"]["labels"])
                feat_offset += len(m.classes_)
                featurizer_data.append(None)
                featureTypes.append("LABEL_BINARIZER")
            elif 'tfidf_vectorizer' in f_rizer and f_rizer["tfidf_vectorizer"] is not None:
                m = TfidfVectorizer(input='content', max_df=config["max_df"], min_df=config["min_df"],
                                    stop_words=stop_words,
                                    decode_error='ignore',
                                    sublinear_tf=config["tf"] == "SUBLINEAR",
                                    smooth_idf=config["df"] == "SMOOTH",
                                    ngram_range = ngram_range, tokenizer = tokenizer)
                m.vocabulary_ = dict()
                for fmap in f_rizer["tfidf_vectorizer"]["vocab"]:
                    m.vocabulary_.update({fmap["term"]: fmap["featureIdx"]})
                m.idf_ = f_rizer["tfidf_vectorizer"]["idf"]
                m.stop_words_ = f_rizer["tfidf_vectorizer"]["stopwords"]
                feat_offset += len(m.vocabulary_)
                featurizer_data.append(dict((fidx, w) for (w, fidx) in m.vocabulary_.items()))
                featureTypes.append("TFIDF_VECTORIZER")
            elif 'multilabel_binarizer' in f_rizer and f_rizer["multilabel_binarizer"] is not None:
                m = MultiLabelBinarizer()
                #must re-fit
                m.fit([f_rizer["multilabel_binarizer"]["labels"]])
                feat_offset += len(m.classes_)
                featurizer_data.append(m.classes_)
                featureTypes.append("MULTILABEL_BINARIZER")
            elif 'label_encoder' in f_rizer and f_rizer["label_encoder"] is not None:
                m = LabelEncoder()
                m.classes_ = np.array(f_rizer["label_encoder"]["labels"])
                feat_offset += 1
                featurizer_data.append(None)
                featureTypes.append("LABEL_ENCODER")
            elif 'doc_to_vector' in f_rizer and f_rizer["doc_to_vector"] is not None:
                m = Doc2Vector(model_file=f_rizer["doc_to_vector"]["modelFile"],
                               max_num_words=f_rizer["doc_to_vector"]["maxNumWords"])
                m.fit()
                feat_offset += m.vector_size()
                featurizer_data.append(None)
                featureTypes.append("TEXT_TO_VECTOR")
            else:
                m = None
                featureTypes.append("NOOP")

            featurizer_models.append(m)
            featurizer_offsets.append(feat_offset)

        featurizer = Featurizer(featureNames, featureTypes,
                        max_df=config["max_df"], min_df=config["min_df"], stop_words=stop_words,
                        sublinear_tf=config["tf"] == "SUBLINEAR", smooth_idf=config["df"] == "SMOOTH",
                        ngram_range=ngram_range, tokenizer=tokenizer)
        featurizer.set_params(models=featurizer_models, model_data=featurizer_data, featurizer_offsets=featurizer_offsets,
                      tokenizer=tokenizer)

        #initialize label encoder model
        m = LabelEncoder()
        m.classes_ = np.array(labelEncoder["labels"])

        labelEncoder = Featurizer(["Label"], ["LABEL"],
                        max_df=config["max_df"], min_df=config["min_df"], stop_words=stop_words,
                        sublinear_tf=config["tf"] == "SUBLINEAR", smooth_idf=config["df"] == "SMOOTH",
                        ngram_range=ngram_range, tokenizer=tokenizer)
        labelEncoder.set_params(models=[m], model_data=[None], featurizer_offsets=[1],
                      tokenizer=tokenizer)
    else:
        clsModel = featurizer = labelEncoder = None

    #debug
    print("loadTrainedModel finished ...")

    return clsModel, featurizer, labelEncoder



def loadModelSelectionResults(json_obj):
    return json.loads(json_obj['json'])



def modelSelectionResultsToObject(objId, msr):
    return {
        'id': objId,
        'json': json.dumps(msr)
    }
