import unittest

from app.logic.helpers import *
from app.test.setup import *

from app.logic.model_selection import model_selection
from app.logic.train import train





modelTypes = [Classifier.LR_MODEL_TYPE, Classifier.SVC_MODEL_TYPE]


class ModelSelectionTest(unittest.TestCase):


    def test_model_selection(self):
        labeled_dataset = random_labeled_dataset()
        model_sel_params = defaultModelSelection()

        trained_models = []
        for _ in range(rd.randint(3, 8)):
            candidate = defaultCandidate(labeled_dataset)[0]
            candidate['config']['type'] = modelTypes[rd.randint(0, len(modelTypes)-1)]
            candidate['config']['C'] = int(rd.random()*100)
            candidate['config']['max_iter'] = 2

            if rd.random() > 0.5:
                #remove a random feature (must not the label)
                rand_fidx = rd.randint(0, len(candidate['features']) - 2)
                candidate['features'] = candidate['features'][:rand_fidx] + candidate['features'][rand_fidx+1:]
                candidate['featurizers'] = candidate['featurizers'][:rand_fidx] + candidate['featurizers'][rand_fidx+1:]

            #randomize configuration
            candidate['config']['penalty'] = "L" + str(rd.randint(1,2))
            candidate['config']['min_df'] = min(rd.random(), .20)
            candidate['config']['max_df'] = max(rd.random(), .80)
            candidate['config']['fitIntercept'] = rd.random() < 0.5
            candidate['config']['weighting'] = 'NONE' if rd.random() < 0.5 else 'BALANCED'
            candidate['config']['tf'] = 'LINEAR' if rd.random() < 0.5 else 'SUBLINEAR'
            candidate['config']['df'] = 'SMOOTH' if rd.random() < 0.5 else 'DEFAULT'

            task = {
                'data': labeled_dataset,
                'candidate': candidate,
                'modelSelectionParams': model_sel_params
            }
            model = train(training_task=task)

            trained_models.append(model)



        for method in ['BEST', 'KNEE_POINT', 'ONE_STDEV', 'TWO_STDEV']:
            model_sel_params['method'] = method

            msr = model_selection(models=trained_models, model_sel=model_sel_params)

            self.assertIn('modelSelection', msr)
            self.assertIn('learnedModels', msr)
            self.assertIn('selectedModel', msr)

            self.assertIn('type', msr['selectedModel'])
            self.assertIn('candidate', msr['selectedModel'])
            self.assertIn('labels', msr['selectedModel'])
            self.assertIn('learnedWeights', msr['selectedModel'])
            self.assertIn('learnedFeaturizers', msr['selectedModel'])
            self.assertIn('labelEncoder', msr['selectedModel'])
            self.assertIn('degreeOfFreedom', msr['selectedModel'])
            self.assertIn('performance', msr['selectedModel'])



if __name__ == "__main__":
    unittest.main()



