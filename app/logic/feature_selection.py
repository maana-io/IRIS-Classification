import pandas as pd
import math

from app.logic.helpers import defaultFeatures, datasetToDataframe, defaultFeaturizers, defaultModelConfiguration
from app.core.main.Classifier import Classifier

from app.core.main.tokenizer.BaseTokenizer import BaseTokenizer
from app.core.main.tokenizer.PorterTokenizer import PorterTokenizer
from app.core.main.tokenizer.LemmaTokenizer import LemmaTokenizer

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS



def top_correlated_features(labeled_dataset, config, topN = None):
    labeled_inp_df = datasetToDataframe(labeled_dataset)
    features = defaultFeatures(dataset=labeled_dataset)
    featurizers = defaultFeaturizers(features)

    stop_words = ENGLISH_STOP_WORDS if config["stopwords"] == "ENGLISH" else []

    tokenizer = BaseTokenizer() if config["tokenizer"] == "WORD_TOKENIZER" \
        else PorterTokenizer() if config["tokenizer"] == "STEMMER" \
        else LemmaTokenizer()  if config["tokenizer"] == "LEMMATIZER" \
        else None

    ngram_range = (1, 1) if config["ngrams"] == "UNIGRAM" \
        else (2, 2) if config["ngrams"] == "BIGRAM" \
        else (1, 2) if config["ngrams"] == "BOTH" \
        else None

    ac = Classifier(model_configuration={
        "type": config['type'],
        "class_weight": config['weighting'].lower(),
        "tokenizer": tokenizer,
        "ngram_range": ngram_range,
        "sublinear_tf": config['tf']=="SUBLINEAR",
        "smooth_idf": config['df']=="SMOOTH",
        "penalty": config['penalty'].lower(),
        "multi_class": config['multiclass'].lower(),
        "solver": config['solver'].lower(),
        "dual": config['primal_dual']=="DUAL",
        "fit_intercept": config['fitIntercept'],
        'max_df':  config['max_df'],
        'min_df':  config['min_df'],
        'stopwords': stop_words,
        'C': config['C'],
        'max_iter': config['max_iter']
    })

    res_df = ac.feature_ranking(input_df=labeled_inp_df, schema=featurizers, mode=Classifier.CC_fs_correlation)

    feature_names = pd.Series(map(lambda fname: fname.split('::')[0], res_df['Feature']))
    feature_scores = pd.concat([feature_names, res_df['Score']], axis=1)
    feature_scores.columns = ['Feature', 'Score']
    feature_sum_scores = feature_scores.groupby('Feature').sum()
    sorted_features = feature_sum_scores.sort_values(by = ["Score"], ascending = False)

    selected_feature_names = list(sorted_features.index)[:topN]
    selected_features = []
    for fname in selected_feature_names:
        selected_features  += [feat for feat in features if feat['name'] == fname]

    return selected_features + [features[-1]]



def top_pct_correlated_features(labeled_dataset, config, pct = 1.):
    features_and_label = top_correlated_features(labeled_dataset, config)
    features = features_and_label[:-1]

    num_features = math.ceil(len(features) * pct)

    return features[:num_features] + [features_and_label[-1]]



def top_rfe_features(labeled_dataset, config, topN = None):
    labeled_inp_df = datasetToDataframe(labeled_dataset)
    features = defaultFeatures(dataset=labeled_dataset)
    featurizers = defaultFeaturizers(features)

    stop_words = ENGLISH_STOP_WORDS if config["stopwords"] == "ENGLISH" else []

    tokenizer = BaseTokenizer() if config["tokenizer"] == "WORD_TOKENIZER" \
        else PorterTokenizer() if config["tokenizer"] == "STEMMER" \
        else LemmaTokenizer()  if config["tokenizer"] == "LEMMATIZER" \
        else None

    ngram_range = (1, 1) if config["ngrams"] == "UNIGRAM" \
        else (2, 2) if config["ngrams"] == "BIGRAM" \
        else (1, 2) if config["ngrams"] == "BOTH" \
        else None


    ac = Classifier(model_configuration={
        "type": config['type'],
        "class_weight": config['weighting'].lower(),
        "tokenizer": tokenizer,
        "ngram_range": ngram_range,
        "sublinear_tf": config['tf']=="SUBLINEAR",
        "smooth_idf": config['df']=="SMOOTH",
        "penalty": config['penalty'].lower(),
        "multi_class": config['multiclass'].lower(),
        "solver": config['solver'].lower(),
        "dual": config['primal_dual']=="DUAL",
        "fit_intercept": config['fitIntercept'],
        'max_df':  config['max_df'],
        'min_df':  config['min_df'],
        'stopwords': stop_words,
        'C': config['C'],
        'max_iter': config['max_iter']
    })

    res_df = ac.feature_ranking(input_df=labeled_inp_df, schema=featurizers, mode=Classifier.CC_fs_backward)

    feature_names = pd.Series(map(lambda fname: fname.split('::')[0], res_df['Feature']))
    feature_scores = pd.concat([feature_names, res_df['Score']], axis=1)
    feature_scores.columns = ['Feature', 'Score']
    feature_sum_scores = feature_scores.groupby('Feature').sum()
    sorted_features = feature_sum_scores.sort_values(by = ["Score"], ascending = False)

    selected_feature_names = list(sorted_features.index)[:topN]
    selected_features = []
    for fname in selected_feature_names:
        selected_features  += [feat for feat in features if feat['name'] == fname]

    return selected_features+ [features[-1]]



def top_pct_rfe_features(labeled_dataset, config, pct = 1.):
    features_and_label = top_rfe_features(labeled_dataset, config)
    features = features_and_label[:-1]

    num_features = math.ceil(len(features) * pct)

    return features[:num_features] + [features_and_label[-1]]
