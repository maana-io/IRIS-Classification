import pandas as pd
import datetime

from app.logic.helpers import datasetToDataframe, dataframeToDataset, id
from app.core.main.Classifier import Classifier
from app.logic.train import loadTrainedModel



cachedModels = {}



def unpackProbs(prob):
    res_dict = {}
    all_labels_list = []

    probList = prob.split(',')
    
    for kv in probList:
        k,v = kv.split(':')
        v = float(v)
        res_dict[k] = v
        predictedLabel = {
            'id': id(),
            'label': k,
            'probability': v
        }
        all_labels_list.append(predictedLabel)
    
    return res_dict, all_labels_list


def unpackContribs(contrib):
    res = []
    if len(contrib) > 0:
        contributors = contrib.split(';')

        for contributor in contributors:
            assert '=' in contributor, "bad contributor:" + '-->' + contributor + '<--' + ' in ' + '"' + contrib + '"'
            feat, weight = contributor.split('=')
            if '::' in feat:
                field_name, field_value = feat.split('::')
            else:
                field_name, field_value = feat, ''

            res.append({
                'id': id(),
                'featureName': field_name,
                'featureValue': field_value,
                'weight': float(weight)
            })

    return res


def unpackSuggestedFeatures(suggestions):
    res = []
    if len(suggestions) > 0:
        suggested_features = suggestions.split(',')

        for feat in suggested_features:
            if '::' in feat:
                field_name, field_value = feat.split('::')
            else:
                field_name, field_value = feat, ''

            res.append({
                'id': id(),
                'featureName': field_name,
                'featureValue': field_value,
                'weight': 1.
            })

    return res



def classify(model, cachedModelID, data):
    startedTime = datetime.datetime.now()
    global cachedModels
    if model is not None:
        cachedModels[cachedModelID] = model
        print('Model cached.')
    elif cachedModelID in cachedModels:
        model = cachedModels[cachedModelID]
        print('Model loaded from cache.')

    assert(model is not None), "No model provided, and there is no cached model."

    print('Model ready time:' + str((datetime.datetime.now() - startedTime).total_seconds()) + ' seconds ')
    startedTime = datetime.datetime.now()


    emptyResults = {
        'id': -1,
        'classSummaries': []
    }

    #debug
    print('Received a dataset with ', len(data['features']), ' features to classify.')
    if (len(data['features']) ==0):
        print('There is no feature, empty result set is returned.')
        return emptyResults
    print('Received a dataset with ', len(data['features'][0]['data']), ' rows to classify.')
    if (len(data['features'][0]['data']) ==0):
        print('There is no data, empty result set is returned.')
        return emptyResults

    candidate = model["candidate"]
    features = candidate["features"]
    config = candidate["config"]

    # #debug
    # print('Model features:', len(features), [f['name'] for f in features])
    # print('Dataset features:', len(data['features']), [f['feature']['name'] for f in data['features']])

    unlabeled_df = datasetToDataframe(data)
    filtered_input_df = unlabeled_df.filter([f['name'] for f in features])

    lr, fm, lm = loadTrainedModel(model)

    ac = Classifier(model_configuration=config)
    ac.load_models(lr, fm, lm)

    res_df = ac.predict_explain(input_df=filtered_input_df, topN_features=10)
    reccom_df = ac.input_qlty(input_df=filtered_input_df, topN=10)
    res_df = pd.concat([res_df, reccom_df.filter(["SuggestedFeatures"])], axis=1)

    plCountSeries = res_df.groupby('PredictedLabel').PredictedLabel.count()
    labels = list(plCountSeries.keys())

    classSummaries = []

    for label in labels:
        filtered_res_df = res_df[res_df.PredictedLabel == label]
        entropies = []
        probabilities = []
        results = []
        for data_index, row in filtered_res_df.iterrows():
            entropies.append(float(row.Entropy))
            probsDict, allLabels = unpackProbs(row.Probabilities)
            probabilities.append(float(probsDict[label]))
            contributors = unpackContribs(row.TopContributors)
            recommends = unpackSuggestedFeatures(row.SuggestedFeatures)

            input_data = []
            for feat in data['features']:
                input_data.append({'id': id(), 'feature': feat['feature'], 'data': [feat['data'][data_index]]})
            data_instance = {
                'id': id(),
                'dataset': { 'id': id(),
                             'features': input_data},
                'index': data_index
            }

            classificationResult = {
                'id': id(),
                'allLabels': allLabels,
                'entropy': float(row.Entropy),
                'contributors': contributors,
                'dataInstance': data_instance,
                'predictedLabel': {
                    'id': id(),
                    'label': label,
                    'probability': float(probsDict[label])
                },
                'recommends': recommends
            }

            results.append(classificationResult)
        
        classSumary = {
            'id': id(),
            'label': label,
            'numInstances': int(plCountSeries[label]),
            'probabilities': probabilities,
            'entropies': entropies,
            'results': results
        }

        classSummaries.append(classSumary)

    batchClassificationResult = {
        'id': id(),
        "classSummaries": classSummaries
    }

    print('Classification time:' + str((datetime.datetime.now() - startedTime).total_seconds()) + ' seconds ')

    return batchClassificationResult
