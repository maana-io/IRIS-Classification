
import pandas as pd

from app.core.main.Classifier import Classifier
from app.core.main.tokenizer.BaseTokenizer import BaseTokenizer
from app.core.main.tokenizer.PorterTokenizer import PorterTokenizer
from app.core.main.tokenizer.LemmaTokenizer import LemmaTokenizer

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

import uuid





def id():
    return str(uuid.uuid4())





def extract(datasource, fields):
    res = {}
    for fld in fields:
        res.update(datasource[fld])
    return res


def getDataFieldName(feature_type):
    field_name = 'text'  # default field name

    if feature_type in ('NUMERICAL', 'SET'):
        field_name = feature_type.lower()
    elif feature_type in ('BOOLEAN'):
        field_name = 'numerical'

    return field_name


def featureDataToSeries(fd):
    feature_name = fd['feature']['name']
    feature_type = fd['feature']['type']
    data = fd['data']

    field_name = getDataFieldName(feature_type)
    rows = [row[field_name] for row in data]
    return pd.Series(rows, name=feature_name)


def datasetToDataframe(ds):
    df = pd.DataFrame()

    features = ds['features'] if 'features' in ds else ds['data']['features']

    for feature in features:
        df = pd.concat([df, featureDataToSeries(feature)], axis=1)

    if 'label' in ds:
        label_series = featureDataToSeries(ds['label'])
        df = pd.concat([df, label_series], axis=1)

    return df



def inferFeatureType(series):
    if series.dtype in ('int64', 'float64'):
        return 'NUMERICAL'

    val = series.values[0]

    if type(val) in (list, set, tuple):
        return 'SET'
    
    val = val.lower()

    if 'isdigit' in val and val.isdigit():
        return 'NUMERICAL'
    if val in ('true', 'false'):
        return 'BOOLEAN'
    
    return 'TEXT'


def dataframeToDataset(df):
    features = []
    
    for col in df:
        feature_index = df.columns.get_loc(col)
        feature_name = col
        feature_type = inferFeatureType(df[col])
        prop_name = getDataFieldName(feature_type)
        data = [{ 'id': id(),
                   prop_name: val } for val in df[col].values]

        feature_data = {
            'id': id(),
            'feature': {
                'id': id(),
                'index': feature_index,
                'name': feature_name,
                'type': feature_type
            },
            'data': data
        }

        features.append(feature_data)

    ds = {
        'id': id(),
        'features': features }

    return ds



def schema(dataset):
    if 'label' in dataset:
        return [f['feature']['type'] for f in dataset['data']['features']] + ['LABEL']
    else:
        return [f['feature']['type'] for f in dataset['features']]




def defaultModelConfiguration():
    return {
        'id': id(),
        "type": "LOGISTIC_REGRESSION",
        "weighting": "BALANCED",
        "tokenizer": "WORD_TOKENIZER",
        "ngrams": "UNIGRAM",
        "tf": "SUBLINEAR",
        "df": "SMOOTH",
        "penalty": "L2",
        "multiclass": "OVR",
        "solver": "LIBLINEAR",
        "primal_dual": "PRIMAL",
        "fitIntercept": True,
        'max_df': 1.,
        'min_df': 0.0,
        'stopwords': "ENGLISH",
        'C': 1.,
        'max_iter': 2,
    }


def defaultModelSelection():
    return {
        'id': id(),
        "metric": "PRECISION",
        "method": "BEST",
        "evalMode": "K_FOLDS",
        "numFolds": 3
    }



def defaultFeaturizers(features):
    feature_types = [f['type'] for f in features]
    featurizer_types = []
    for ft in feature_types:
        if ft in ["NUMERICAL", "BOOLEAN"]:
            featurizer_types.append("NOOP")
        elif ft in ["CATEGORICAL"]:
            featurizer_types.append("LABEL_BINARIZER")
        elif ft in ["TEXT"]:
            featurizer_types.append("TFIDF_VECTORIZER")
        elif ft in ["SET"]:
            featurizer_types.append("MULTILABEL_BINARIZER")
        elif ft == "LABEL":
            featurizer_types.append("LABEL")
        elif ft in ["LABEL_ENCODER"]:
            featurizer_types.append("LABEL_ENCODER")
        elif ft in ["TEXT2VEC", "TEXT_TO_VECTOR"]:
            featurizer_types.append("TEXT_TO_VECTOR")

    return featurizer_types



def defaultCandidate(labeled_dataset):
    features = defaultFeatures(labeled_dataset)
    return [{
            'id': id(),
            "features": features,
            "featurizers": defaultFeaturizers(features=features),
            "config": defaultModelConfiguration()
        }]




def createCandidate(features, featurizers, config):
    return {
        'id': id(),
        "features": features,
        "featurizers": featurizers,
        "config": config
    }



def createTrainingTasks(candidates, training_data, params):
    return [{
        'id': id(),
        'candidate': candidate,
        'data': training_data,
        'modelSelectionParams': params
    } for candidate in candidates]




def createFeaturizer(featureType, featurizerType):
    return {
        'id': id(),
        "featureType": featureType,
        "featurizer": featurizerType
    }



def createModelConfiguration(
    type, weighting, tokenizer, ngrams, tf,
    df, penalty, multiclass, solver, primal_dual,
    fitIntercept, max_df, min_df, stopwords, C=1, max_iter=2
):
    return {
        'id': id(),
        "type": type,
        "weighting": weighting,
        "tokenizer": tokenizer,
        "ngrams": ngrams,
        "tf": tf,
        "df": df,
        "penalty": penalty,
        "multiclass": multiclass,
        "solver": solver,
        "primal_dual": primal_dual,
        "fitIntercept": fitIntercept,
        "max_df": max_df,
        "min_df": min_df,
        "stopwords": stopwords,
        "C": C,
        "max_iter": max_iter
    }


def createModelSelection(metric, method, evalMode, numFolds):
    return {
        'id': id(),
        "metric": metric,
        "method": method,
        "evalMode": evalMode,
        "numFolds": numFolds
    }


def defaultFeatures(dataset):
    if 'label' in dataset:
        feature_data = dataset['data']['features'] + [dataset['label']]
    else:
        feature_data = dataset['features']
    features = [ feat_dat['feature'] for feat_dat in feature_data]

    return features


def subsetFeatures(dataset, selectedFeatures):
    allFeatures = defaultFeatures(dataset)
    return [allFeatures[i] for i in selectedFeatures]






def create_classifier(config):

    return Classifier(model_configuration={
        'id': id(),
        "type": config["type"],
        "class_weight": None if config['weighting'].lower() == 'none' else config['weighting'].lower(),
        "tokenizer": BaseTokenizer() if config["tokenizer"] == "WORD_TOKENIZER" \
            else PorterTokenizer() if config["tokenizer"] == "STEMMER" \
            else LemmaTokenizer()  if config["tokenizer"] == "LEMMATIZER" \
            else None,
        "ngram_range": (1, 1) if config["ngrams"] == "UNIGRAM" \
            else (2, 2) if config["ngrams"] == "BIGRAM" \
            else (1, 2) if config["ngrams"] == "BOTH" \
            else None,
        "sublinear_tf": config["tf"] == "SUBLINEAR",
        "smooth_idf": config["df"] == "SMOOTH",
        "penalty": config['penalty'].lower(),
        "multi_class": config['multiclass'].lower(),
        "solver": config['solver'].lower(),
        "dual": config['primal_dual']=='DUAL',
        "fit_intercept": config['fitIntercept'],
        'max_df': config['max_df'],
        'min_df': config['min_df'],
        'stopwords': ENGLISH_STOP_WORDS if config["stopwords"] == "ENGLISH" else [],
        'C': config['C'],
        'max_iter': config['max_iter']
    })



def addToList(addThis, toThis):
    if toThis is None:
        return [addThis]
    else:
        toThis.append(addThis)
        return toThis
