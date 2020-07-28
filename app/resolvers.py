
import logging
from app.logic.helpers import *
from app.logic.active_learning import *
from app.logic.classify import *
from app.logic.feature_selection import *
from app.logic.model_selection import *
from app.logic.train import *

logger = logging.getLogger(__name__)

resolvers = {
    'Query': {
        #'train': lambda value, info, **args: train(args['trainingTask']),
        'train_batch': lambda value, info, **args: train_batch(args['candidates'], args['training_data'],
            args['params'], args['model_id']),
        'get_training_results': lambda value, info, **args: get_training_results(args['model_id']),
        'delete_training_results': lambda value, info, **args: delete_training_results(args['model_id']),
        #'modelSelection': lambda value, info, **args: model_selection(args['models'], args['modelSel']),
        'classify': lambda value, info, **args: classify(args['cachedModelID'], args['data']),

        'loadModelSelectionResults': lambda value, info, **args: loadModelSelectionResults(args['obj']),
        'modelSelectionResultsToObject': lambda value, info, **args: modelSelectionResultsToObject(args['savedId'], args['msr']),

        'defaultModelConfiguration': lambda value, info, **args: defaultModelConfiguration(),
        'defaultModelSelection': lambda value, info, **args: defaultModelSelection(),
        'defaultFeatures': lambda value, info, **args:  defaultFeatures(args['dataset']),
        'defaultFeaturizers': lambda value, info, **args:  defaultFeaturizers(args['features']),
        'defaultCandidate': lambda value, info, **args:  defaultCandidate(args['dataset']),
        'createModelConfiguration':  lambda value, info, **args: createModelConfiguration(
            args['type'], args['weighting'], args['tokenizer'], args['ngrams'], args['tf'],
            args['df'], args['penalty'], args['multiclass'], args['solver'], args['primal_dual'],
            args['fitIntercept'], args['max_df'], args['min_df'], args['stopwords'], args['C'], args['max_iter']
        ),
        'createModelSelection':  lambda value, info, **args: createModelSelection(
            args['metric'], args['method'], args['evalMode'], args['numFolds']
        ),
        'createCandidate':  lambda value, info, **args: createCandidate(args['features'], args['featurizers'], args['config']),
        'addCandidate':  lambda value, info, **args: addToList(args['addThis'], args.get('toThis', None)),
        #'createTrainingTasks': lambda value, info, **args:  createTrainingTasks(args['candidates'],
        #        args['training_data'], args['params']),

        'addFeaturizer':  lambda value, info, **args: addToList(args['addThis'], args.get('toThis', None)),
        'subsetFeatures': lambda value, info, **args:  subsetFeatures(args['dataset'], args['selectedFeatures']),
        'topNCorrelatedFeatures': lambda value, info, **args:  top_correlated_features(args['dataset'], args['config'], args['topN']),
        'topNPctCorrelatedFeatures': lambda value, info, **args:  top_pct_correlated_features(args['dataset'], args['config'], args['pct']),
        'topNRFEFeatures': lambda value, info, **args:  top_rfe_features(args['dataset'], args['config'], args['topN']),
        'topNPctRFEFeatures': lambda value, info, **args:  top_pct_rfe_features(args['dataset'], args['config'], args['pct']),

        'numericalFeatureType': lambda value, info, **args: 'NUMERICAL',
        'categoricalFeatureType': lambda value, info, **args: 'CATEGORICAL',
        'textFeatureType': lambda value, info, **args: 'TEXT',
        'setFeatureType': lambda value, info, **args: 'SET',
        'booleanFeatureType': lambda value, info, **args: 'BOOLEAN',
        'labelFeatureType': lambda value, info, **args: 'LABEL',

        'noopFeaturizerType': lambda value, info, **args: 'NOOP',
        'minMaxScalerFeaturizerType': lambda value, info, **args: 'MIN_MAX_SCALER',
        'labelBinarizerFeaturizerType': lambda value, info, **args: 'LABEL_BINARIZER',
        'tfidfVectorizerFeaturizerType': lambda value, info, **args: 'TFIDF_VECTORIZER',
        'multilabelBinarizerFeaturizerType': lambda value, info, **args: 'MULTILABEL_BINARIZER',
        'labelEncoderFeaturizerType': lambda value, info, **args: 'LABEL',
        'textToVectorFeaturizerType': lambda value, info, **args: 'TEXT_TO_VECTOR',
        
        'correlationFeatureSelectionMode': lambda value, info, **args: 'CORRELATION',
        'rfeFeatureSelectionMode': lambda value, info, **args: 'RFE',
        
        'logisticRegressionModelType': lambda value, info, **args: 'LOGISTIC_REGRESSION',
        'linearSVCModelType': lambda value, info, **args: 'LINEAR_SVC',
        
        'noneClassWeightingType': lambda value, info, **args: 'NONE',
        'balancedClassWeightingType': lambda value, info, **args: 'BALANCED',
        
        'unigramNGramType': lambda value, info, **args: 'UNIGRAM',
        'bigramNGramType': lambda value, info, **args: 'BIGRAM',
        'bothNGramType': lambda value, info, **args: 'BOTH',
        
        'linearTermFreqType': lambda value, info, **args: 'LINEAR',
        'sublinearTermFreqType': lambda value, info, **args: 'SUBLINEAR',
        
        'defaultDocumentFreqType': lambda value, info, **args: 'DEFAULT',
        'smoothDocumentFreqType': lambda value, info, **args: 'SMOOTH',
        
        'l1PenaltyType': lambda value, info, **args: 'L1',
        'l2PenaltyType': lambda value, info, **args: 'L2',
        
        'ovrMultiClassType': lambda value, info, **args: 'OVR',
        'multinomialMultiClassType': lambda value, info, **args: 'MULTINOMIAL',
        'autoMultiClassType': lambda value, info, **args: 'AUTO',
        
        'liblinearSolverType': lambda value, info, **args: 'LIBLINEAR',
        'newtonCGSolverType': lambda value, info, **args: 'NEWTON_CG',
        'lbfgsSolverType': lambda value, info, **args: 'LBFGS',
        'sagSolverType': lambda value, info, **args: 'SAG',
        'sagaSolverType': lambda value, info, **args: 'SAGA',
        
        'primalMode': lambda value, info, **args: 'PRIMAL',
        'dualMode': lambda value, info, **args: 'DUAL',
        
        'wordTokenizerType': lambda value, info, **args: 'WORD_TOKENIZER',
        'stemmerTokenizerType': lambda value, info, **args: 'STEMMER',
        'lemmatizerTokenizerType': lambda value, info, **args: 'LEMMATIZER',
        
        'precisionModelSelectionMetric': lambda value, info, **args: 'PRECISION',
        'recallModelSelectionMetric': lambda value, info, **args: 'RECALL',
        'f1ModelSelectionMetric': lambda value, info, **args: 'F1',
        
        'bestModelModelSelectionMethod': lambda value, info, **args: 'BEST',
        'kneePointModelSelectionMethod': lambda value, info, **args: 'KNEE_POINT',
        'oneStdevModelSelectionMethod': lambda value, info, **args: 'ONE_STDEV',
        'twoStdevModelSelectionMethod': lambda value, info, **args: 'TWO_STDEV',
        
        'looModelEvaluationType': lambda value, info, **args: 'LEAVE_ONE_OUT',
        'kFoldsModelEvaluationType': lambda value, info, **args: 'K_FOLDS',
        
        'noneStopWordType': lambda value, info, **args: 'NONE',
        'englishStopWordType': lambda value, info, **args: 'ENGLISH',
    },
    'Object': {
    },
    'Mutation': {
    },
    'Scalar': {
    },
}





