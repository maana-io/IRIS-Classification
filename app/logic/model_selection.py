
import sys
import numpy as np

#import pprint
from app.logic.train import evaluate, train, id

#pp = pprint.PrettyPrinter(indent=3)

import logging
from app.settings import LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL)


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


