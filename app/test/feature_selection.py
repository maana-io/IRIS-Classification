import unittest
import sys

from app.test.setup import *

from app.logic.feature_selection import *




modelTypes = [Classifier.LR_MODEL_TYPE, Classifier.SVC_MODEL_TYPE, Classifier.ENSEMBLE_LR_MODEL_TYPE, Classifier.ENSEMBLE_SVC_MODEL_TYPE]

testModelType = modelTypes[0]



class FeatureSelectionTest(unittest.TestCase):

    def test_topN_correlation(self):
        labeled_dataset = random_labeled_dataset()
        num_top_features = rd.randint(10, 100)
        num_top_features = min(num_top_features, len(labeled_dataset['data']['features']))

        config = defaultModelConfiguration()
        config['type'] = testModelType
        config['C'] = 10.
        config['max_iter'] = 2

        #incl label
        ranked_features = top_correlated_features(labeled_dataset, config, topN=num_top_features)

        self.assertLessEqual(len(ranked_features), num_top_features + 1)
        self.assertIn('id', ranked_features[0])
        self.assertIn('index', ranked_features[0])
        self.assertIn('name', ranked_features[0])
        self.assertIn('type', ranked_features[0])

    def test_topN_pct_correlation(self):
        labeled_dataset = random_labeled_dataset()
        pct_top_features = rd.random()
        config = defaultModelConfiguration()
        config['type'] = testModelType
        config['C'] = 10.
        config['max_iter'] = 2


        ranked_features = top_pct_correlated_features(labeled_dataset, config, pct=pct_top_features)

        self.assertIn('id', ranked_features[0])
        self.assertIn('index', ranked_features[0])
        self.assertIn('name', ranked_features[0])
        self.assertIn('type', ranked_features[0])




    def test_topN_backward(self):
        if testModelType in [Classifier.ENSEMBLE_SVC_MODEL_TYPE, Classifier.ENSEMBLE_LR_MODEL_TYPE]:
            return

        labeled_dataset = random_labeled_dataset()
        num_top_features = rd.randint(10, 100)
        config = defaultModelConfiguration()
        config['type'] = testModelType
        config['C'] = 10.
        config['max_iter'] = 2

        ranked_features = top_rfe_features(labeled_dataset, config, topN=num_top_features)

        self.assertLessEqual(len(ranked_features), num_top_features+1)      #label is always included
        self.assertIn('id', ranked_features[0])
        self.assertIn('index', ranked_features[0])
        self.assertIn('name', ranked_features[0])
        self.assertIn('type', ranked_features[0])




    def test_topN_pct_backward(self):
        if testModelType in [Classifier.ENSEMBLE_SVC_MODEL_TYPE, Classifier.ENSEMBLE_LR_MODEL_TYPE]:
            return

        labeled_dataset = random_labeled_dataset()
        pct_top_features = rd.random()
        config = defaultModelConfiguration()
        config['type'] = testModelType
        config['C'] = 10.
        config['max_iter'] = 2

        ranked_features = top_pct_rfe_features(labeled_dataset, config, pct=pct_top_features)

        self.assertIn('id', ranked_features[0])
        self.assertIn('index', ranked_features[0])
        self.assertIn('name', ranked_features[0])
        self.assertIn('type', ranked_features[0])


if __name__ == '__main__':
    testModelType = modelTypes[int(sys.argv[1])]
    print("Testing ", testModelType)
    unittest.main(argv=[''])
