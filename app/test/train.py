import unittest
import sys

from app.logic.helpers import *
from app.test.setup import *

from app.logic.train import evaluate, train, loadTrainedModel

from app.core.main.Classifier import Classifier



modelTypes = [Classifier.LR_MODEL_TYPE, Classifier.SVC_MODEL_TYPE, Classifier.ENSEMBLE_LR_MODEL_TYPE, Classifier.ENSEMBLE_SVC_MODEL_TYPE]
testModelType = modelTypes[0]


class TrainingTest(unittest.TestCase):

    def test_train(self):
        labeled_dataset = random_labeled_dataset()
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

        self.assertIn('type', model)
        self.assertIn('candidate', model)
        self.assertIn('labels', model)
        self.assertIn('learnedWeights', model)
        self.assertIn('learnedFeaturizers', model)
        self.assertIn('labelEncoder', model)
        self.assertIn('degreeOfFreedom', model)
        self.assertIn('performance', model)



    def test_evaluate(self):
        labeled_dataset = random_labeled_dataset()
        candidate = defaultCandidate(labeled_dataset)[0]
        model_sel_params = defaultModelSelection()

        task = {
            'data': labeled_dataset,
            'candidate': candidate,
            'modelSelectionParams': model_sel_params
        }

        model_performance = evaluate(task)

        self.assertIn('classPerformances', model_performance)
        self.assertIn('numInstances', model_performance)
        self.assertIn('avgRecall', model_performance)
        self.assertIn('avgPrecision', model_performance)
        self.assertIn('avgF1', model_performance)



if __name__ == "__main__":
    testModelType = modelTypes[int(sys.argv[1])]
    print("Testing ", testModelType)
    unittest.main(argv=[''])



