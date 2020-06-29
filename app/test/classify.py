# coding: utf-8
import unittest
import sys

from app.logic.helpers import *
from app.test.setup import *

from app.logic.classify import classify
from app.logic.train import *


modelTypes = [Classifier.LR_MODEL_TYPE, Classifier.SVC_MODEL_TYPE, Classifier.ENSEMBLE_LR_MODEL_TYPE, Classifier.ENSEMBLE_SVC_MODEL_TYPE]

testModelType = modelTypes[0]



class ClassifyTest(unittest.TestCase):

    def test_classify(self):
        labeled_dataset = random_labeled_dataset()

        lbl_dataset_id = labeled_dataset['id']

        candidate = defaultCandidate(labeled_dataset)[0]
        candidate['config']['type'] = testModelType
        candidate['config']['C'] = 10
        candidate['config']['max_iter'] = 2

        model_sel_params = defaultModelSelection()

        task = {
            'data': labeled_dataset,
            'candidate': candidate,
            'modelSelectionParams': model_sel_params
        }
        model = train(training_task=task)

        test_set = {
            'id': id(),
            'features': labeled_dataset['data']['features']
        }

        batch_classification_res = classify(model=model, data=test_set)

        self.assertTrue(isinstance(batch_classification_res, dict))
        self.assertIn('classSummaries', batch_classification_res)

        self.assertTrue(isinstance(batch_classification_res['classSummaries'], list))
        self.assertTrue(isinstance(batch_classification_res['classSummaries'][0], dict))
        self.assertIn('label', batch_classification_res['classSummaries'][0])
        self.assertIn('numInstances', batch_classification_res['classSummaries'][0])
        self.assertIn('probabilities', batch_classification_res['classSummaries'][0])
        self.assertIn('entropies', batch_classification_res['classSummaries'][0])
        self.assertIn('results', batch_classification_res['classSummaries'][0])


if __name__ == '__main__':
    testModelType = modelTypes[int(sys.argv[1])]
    print("Testing ", testModelType)
    unittest.main(argv=[''])
