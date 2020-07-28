
import sys
import numpy as np
import datetime
import logging
from pathlib import Path
import pickle
import os

from app.logic.train import *
from app.logic.helpers import *
from app.settings import *

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL)



cachedMSR = {}



def model_selection(models, model_sel):
    degree_of_freedom =  np.array([model['degreeOfFreedom'] for model in models])

    if model_sel['metric']== 'RECALL':
        metrics = np.array([model['performance']['avgRecall'] for model in models])
    elif model_sel['metric'] == 'PRECISION':
        metrics = np.array([model['performance']['avgPrecision'] for model in models])
    elif model_sel['metric'] == 'F1':
        metrics = np.array([model['performance']['avgF1'] for model in models])

    if model_sel['method']== 'BEST':
        selected_midx = np.argmax(metrics)
    elif model_sel['method']== 'KNEE_POINT':
        selected_midx = knee_point(metrics, degree_of_freedom)
    elif model_sel['method']== 'ONE_STDEV':
        selected_midx = one_stdev(metrics, degree_of_freedom)
    elif model_sel['method']== 'TWO_STDEV':
        selected_midx = two_stdev(metrics, degree_of_freedom)
    else:
        selected_midx = 0

    selected_model = models[selected_midx]

    #type ModelSelectionResults
    res = {
        'id': id(),
        'modelSelection': model_sel,
        'learnedModels': models,
        'selectedModel': selected_model
    }

    return res



def knee_point(metrics, degree_of_freedom):
    num_models = len(metrics)

    if num_models == 1:
        opt_split_idx = 0
    else:
        metrics_with_dof = zip(metrics, degree_of_freedom, range(num_models))

        sorted_metrics_by_dof = sorted(metrics_with_dof, key = lambda metric_dof_idx: -metric_dof_idx[1])
        err = np.zeros(num_models - 1, dtype=np.float)
        for split_idx in range(num_models - 1):
            left_ = np.array([m for (m, _, _) in sorted_metrics_by_dof[:split_idx+1]])
            right_ = np.array([m for (m, _, _) in sorted_metrics_by_dof[split_idx+1:]])
            err1 = 0 if len(left_) < 2 else sum(abs(left_ - np.average(left_)))
            err2 = 0 if len(right_) < 2 else sum(abs(right_ - np.average(right_)))
            err[split_idx] = err1 + err2

        opt_split_idx = np.argmin(err)
    return opt_split_idx




def one_stdev(metrics, degree_of_freedom):
    num_models = len(metrics)
    metrics_with_dof = zip(metrics, degree_of_freedom, range(num_models))

    avg = np.average(metrics)
    std = np.std(metrics)
    lower_bound =  avg - std
    upper_bound = avg + std

    eligible = [ mm for mm in metrics_with_dof if mm[0] >= lower_bound and mm[0] <= upper_bound ]

    lowest_dof_idx = np.argmin([ mm[1] for mm in eligible ])
    opt_idx = eligible[lowest_dof_idx][2]

    return opt_idx




def two_stdev(metrics, degree_of_freedom):
    num_models = len(metrics)
    metrics_with_dof = zip(metrics, degree_of_freedom, range(num_models))

    avg = np.average(metrics)
    std = np.std(metrics)
    lower_bound =  avg - 2*std
    upper_bound = avg + 2*std

    eligible = [ mm for mm in metrics_with_dof if mm[0] >= lower_bound and mm[0] <= upper_bound ]

    lowest_dof_idx = np.argmin([ mm[1] for mm in eligible ])
    opt_idx = eligible[lowest_dof_idx][2]

    return opt_idx



def train_batch(candidates, training_data, model_selection_params, model_id):
    startedTime = datetime.datetime.now()
    global cachedMSR

    training_tasks = createTrainingTasks(candidates, training_data, model_selection_params)
    trained_models = list(map(train, training_tasks))
    msr = model_selection(trained_models, model_selection_params)
    msr['id'] = model_id

    cachedMSR[model_id] = msr
    save_training_results(model_id)

    seconds = (datetime.datetime.now() - startedTime).total_seconds()
    print('Trained ' + str(len(training_tasks)) + ' models in ' + str(seconds//60) + ' minutes ' + str(seconds%60) + ' seconds.')
    print('Model ' + str(model_id) + ' cached.')

    return msr



def delete_training_results(model_id):
    assert(model_id in cachedMSR), 'Model ID ' + str(model_id) + ' not found.'
    cachedMSR.pop(model_id)
    remove_model(model_id)
    return model_id



def get_training_results(model_id):
    global cachedMSR

    assert(model_id in cachedMSR), 'Training results with given ID not found.'
    return cachedMSR[model_id]



def save_training_results(model_id):
    global cachedMSR

    assert(model_id in cachedMSR), 'Training results with given ID not found.'
    path = Path(CLASSIFICATION_DATA_DIR)
    path.mkdir(parents=True, exist_ok=True)
    output = open(CLASSIFICATION_DATA_DIR + "/" + model_id + ".pkl", 'wb')
    pickle.dump({model_id: cachedMSR[model_id]}, output)
    output.close()
    print("Model " + str(model_id) + " saved.")



def load_models():
    global cachedMSR

    path = Path(CLASSIFICATION_DATA_DIR)
    path.mkdir(parents=True, exist_ok=True)
    fileNames = os.listdir(CLASSIFICATION_DATA_DIR)
    print("Loading models from " + CLASSIFICATION_DATA_DIR)
    for fname in fileNames:
        modelDataFile = Path(CLASSIFICATION_DATA_DIR + "/" + fname)
        if modelDataFile.is_file():
            datafile = open(CLASSIFICATION_DATA_DIR + "/" + fname, 'rb')
            modelData = pickle.load(datafile)
            cachedMSR.update(modelData)
            datafile.close()
            print("Model " + fname + " loaded.")
    print(str(len(cachedMSR)) + " models loaded from " + CLASSIFICATION_DATA_DIR)
    return cachedMSR



def remove_model(model_id):
    filename = CLASSIFICATION_DATA_DIR + "/" + model_id + ".pkl"
    modelData = Path(filename)
    if modelData.is_file():
        os.remove(filename)
        print("Model " + str(model_id) + " deleted.")
    else:
        print("Model " + str(model_id) + " not found.")



def all_training_results():
    return list(cachedMSR.values())




cachedMSR = load_models()
