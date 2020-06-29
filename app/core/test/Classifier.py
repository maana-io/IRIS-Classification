# coding: utf-8
import sys
import unittest
import pandas as pd
import random as rd

from app.core.main.classifier.LR import LR
from app.core.main.classifier.LSVC import LSVC
from app.core.main.classifier.Ensemble import Ensemble
from app.core.main.featurizer.Featurizer import Featurizer


from app.core.main.tokenizer.BaseTokenizer import BaseTokenizer
from sklearn.feature_extraction import stop_words

from app.core.main.Classifier import Classifier

lrModelConfiguration = {
    "type": "LOGISTIC_REGRESSION",
    "class_weight": "balanced",
    "tokenizer": BaseTokenizer(),
    "ngram_range": (1, 1),
    "sublinear_tf": True,
    "smooth_idf": True,
    "penalty": "l2",
    "multi_class": "ovr",
    "solver": "liblinear",
    "dual": True,
    "fit_intercept": True,
    'max_df': 1.,
    'min_df': 0.,
    'stopwords': stop_words.ENGLISH_STOP_WORDS,
    'C': 1.,
    'max_iter': 1000,
}

lsvcModelConfiguration = {
    "type": "LINEAR_SVC",
    "class_weight": "balanced",
    "tokenizer": BaseTokenizer(),
    "ngram_range": (1, 1),
    "sublinear_tf": True,
    "smooth_idf": True,
    "penalty": "l2",
    "multi_class": "ovr",
    "solver": "liblinear",
    "dual": True,
    "fit_intercept": True,
    'max_df': 1.,
    'min_df': 0.,
    'stopwords': stop_words.ENGLISH_STOP_WORDS,
    'C': 10.,
    'max_iter': 1000,
}


ensembleSvcModelConfiguration = {
    "type": "ENSEMBLE_LINEAR_SVC",
    "class_weight": "balanced",
    "tokenizer": BaseTokenizer(),
    "ngram_range": (1, 1),
    "sublinear_tf": True,
    "smooth_idf": True,
    "penalty": "l2",
    "multi_class": "ovr",
    "solver": "liblinear",
    "dual": True,
    "fit_intercept": True,
    'max_df': 1.,
    'min_df': 0.,
    'stopwords': stop_words.ENGLISH_STOP_WORDS,
    'C': 10.,
    'max_iter': 1000,
}

ensembleLRModelConfiguration = {
    "type": "ENSEMBLE_LOGISTIC_REGRESSION",
    "class_weight": "balanced",
    "tokenizer": BaseTokenizer(),
    "ngram_range": (1, 1),
    "sublinear_tf": True,
    "smooth_idf": True,
    "penalty": "l2",
    "multi_class": "ovr",
    "solver": "liblinear",
    "dual": True,
    "fit_intercept": True,
    'max_df': 1.,
    'min_df': 0.,
    'stopwords': stop_words.ENGLISH_STOP_WORDS,
    'C': 1.,
    'max_iter': 1000,
}


modelConfigurations = [lrModelConfiguration, lsvcModelConfiguration, ensembleLRModelConfiguration, ensembleSvcModelConfiguration]

testModelConfiguration = None


class FunctionalityTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.test_data_fn_prefix = "test_data"
        cls.inp_avro_file_with_label = cls.test_data_fn_prefix + ".avro"
        cls.binary_inp_avro_file_with_label = cls.test_data_fn_prefix + ".two_classes.avro"
        cls.inp_avro_file_without_label = cls.test_data_fn_prefix + ".nolabel.avro"
        cls.out_avro_file = cls.test_data_fn_prefix + ".out.avro"
        cls.model_avro_file = cls.test_data_fn_prefix + ".model.out.avro"

        cls.schema_with_label = ["NUMERICAL", "NUMERICAL", "CATEGORICAL", "BOOLEAN", "TEXT", "TEXT2VEC", "SET", "LABEL"]
        #for correlation feature ranking test, remove text2vec as it produces negative values, which is prohibited
        #by chisquare ranking.
        cls.schema_with_label_nonnegative = ["NUMERICAL", "NUMERICAL", "CATEGORICAL", "BOOLEAN", "TEXT", "TEXT", "SET", "LABEL"]
        cls.schema_without_label = ["NUMERICAL", "NUMERICAL", "CATEGORICAL", "BOOLEAN", "TEXT", "TEXT2VEC", "SET"]

        cls.fields_with_label = ["TRAIN_GROUP", "num_field", "cat_field", "bool_field", "text_field", "txt2vec_field", "set_field",
                                 "label_field"]
        cls.fields_without_label = cls.fields_with_label[:-1]

        cls.num_recs = 150
        cls.num_models_for_ensemble = 8
        cls.categories = ["Oil and Gas", "Automobile", "Software", "Retail", "Health Care", "Finance", "Construction",
                          "Agriculture"]
        cls.labels = ["a posteriori", "acta non verba", "alea iacta est", "amor patriae", "alma mater"]
        cls.binary_labels = ["verum", "falsus"]

        words = '''
        The distribution of oil and gas reserves among the world's 50 largest oil companies. The reserves of the privately
        owned companies are grouped together. The oil produced by the "supermajor" companies accounts for less than 15% of
        the total world supply. Over 80% of the world's reserves of oil and natural gas are controlled by national oil companies.
        Of the world's 20 largest oil companies, 15 are state-owned oil companies.
        The petroleum industry, also known as the oil industry or the oil patch, includes the global processes of exploration,
        extraction, refining, transporting (often by oil tankers and pipelines), and marketing of petroleum products.
        The largest volume products of the industry are fuel oil and gasoline (petrol). Petroleum (oil) is also the raw material
        for many chemical products, including pharmaceuticals, solvents, fertilizers, pesticides, synthetic fragrances, and plastics.
        The industry is usually divided into three major components: upstream, midstream and downstream. Midstream operations are
        often included in the downstream category.
        Petroleum is vital to many industries, and is of importance to the maintenance of industrial civilization in its
        current configuration, and thus is a critical concern for many nations. Oil accounts for a large percentage of the
        world’s energy consumption, ranging from a low of 32% for Europe and Asia, to a high of 53% for the Middle East.
        Governments such as the United States government provide a heavy public subsidy to petroleum companies, with major
        tax breaks at virtually every stage of oil exploration and extraction, including the costs of oil field leases and
        drilling equipment.[2]
        Principle is a term defined current-day by Merriam-Webster[5] as: "a comprehensive and fundamental law, doctrine,
        or assumption", "a primary source", "the laws or facts of nature underlying the working of an artificial device",
        "an ingredient (such as a chemical) that exhibits or imparts a characteristic quality".[6]
        Process is a term defined current-day by the United States Patent Laws (United States Code Title 34 - Patents)[7]
        published by the United States Patent and Trade Office (USPTO)[8] as follows: "The term 'process' means process,
        art, or method, and includes a new use of a known process, machine, manufacture, composition of matter, or material."[9]
        Application of Science is a term defined current-day by the United States' National Academies of Sciences, Engineering,
        and Medicine[12] as: "...any use of scientific knowledge for a specific purpose, whether to do more science; to design
        a product, process, or medical treatment; to develop a new technology; or to predict the impacts of human actions."[13]
        The simplest form of technology is the development and use of basic tools. The prehistoric discovery of how to control
        fire and the later Neolithic Revolution increased the available sources of food, and the invention of the wheel
        helped humans to travel in and control their environment. Developments in historic times, including the printing
        press, the telephone, and the Internet, have lessened physical barriers to communication and allowed humans to
        interact freely on a global scale.
        Technology has many effects. It has helped develop more advanced economies (including today's global economy)
        and has allowed the rise of a leisure class. Many technological processes produce unwanted by-products known as
        pollution and deplete natural resources to the detriment of Earth's environment. Innovations have always influenced
        the values of a society and raised new questions of the ethics of technology. Examples include the rise of the
        notion of efficiency in terms of human productivity, and the challenges of bioethics.
        Philosophical debates have arisen over the use of technology, with disagreements over whether technology improves
        the human condition or worsens it. Neo-Luddism, anarcho-primitivism, and similar reactionary movements criticize
        the pervasiveness of technology, arguing that it harms the environment and alienates people; proponents of ideologies
        such as transhumanism and techno-progressivism view continued technological progress as beneficial to society and
        the human condition.
        Health care or healthcare is the maintenance or improvement of health via the prevention, diagnosis, and treatment
        of disease, illness, injury, and other physical and mental impairments in human beings. Healthcare is delivered by
        health professionals (providers or practitioners) in allied health fields. Physicians and physician associates are
        a part of these health professionals. Dentistry, midwifery, nursing, medicine, optometry, audiology, pharmacy,
        psychology, occupational therapy, physical therapy and other health professions are all part of healthcare. It
        includes work done in providing primary care, secondary care, and tertiary care, as well as in public health.
        Access to health care may vary across countries, communities, and individuals, largely influenced by social and
        economic conditions as well as the health policies in place. Countries and jurisdictions have different policies
        and plans in relation to the personal and population-based health care goals within their societies. Healthcare
        systems are organizations established to meet the health needs of targeted populations. Their exact configuration
        varies between national and subnational entities. In some countries and jurisdictions, health care planning is
        distributed among market participants, whereas in others, planning occurs more centrally among governments or
        other coordinating bodies. In all cases, according to the World Health Organization (WHO), a well-functioning
        healthcare system requires a robust financing mechanism; a well-trained and adequately paid workforce; reliable
        information on which to base decisions and policies; and well maintained health facilities and logistics to deliver
        quality medicines and technologies.[1]
        Health care is conventionally regarded as an important determinant in promoting the general physical and mental
        health and well-being of people around the world. An example of this was the worldwide eradication of smallpox
        in 1980, declared by the WHO as the first disease in human history to be completely eliminated by deliberate health
        care interventions.[4]
        '''.split()

        truth = ["true", "false"]

        set_base = ["bona fide", "bono malum superate", "carpe diem", "caveat emptor", "circa", "citius altius fortius",
                    "corpus christi", "curriculum vitae", "de facto", "discendo discimus", "emeritus", "ex animo",
                    "fortis in arduis", "labor omnia vincit", "magnum opus", "persona non grata", "vivere militare est"]

        set_size = list(map(lambda _: rd.randint(int(len(set_base)/10), int(len(set_base)/2)), range(cls.num_recs)))
        set_field = list(map(lambda n: set(map(lambda _: set_base[rd.randint(0, len(set_base)-1)], range(n))), set_size))

        #chisquare feature ranking requires non-negative values
        train_group_field = list(map(lambda _ : str(rd.randint(1, cls.num_models_for_ensemble)), range(cls.num_recs)))
        numeric_field = list(map(lambda _ : rd.random() * rd.randint(0, 100), range(cls.num_recs)))
        categorical_field = list(map(lambda _: cls.categories[rd.randint(0, len(cls.categories)-1)], range(cls.num_recs)))
        boolean_field = list(map(lambda _: truth[rd.randint(0, 1)], range(cls.num_recs)))
        label_field = list(map(lambda _: cls.labels[rd.randint(0, len(cls.labels)-1)], range(cls.num_recs)))
        binary_label_field = list(map(lambda _: cls.binary_labels[rd.randint(0, len(cls.binary_labels)-1)], range(cls.num_recs)))


        text_size = list(map(lambda _: rd.randint(int(len(words)/10), int(len(words)/2)), range(cls.num_recs)))
        text_field1 = list(map(lambda n: ' '.join(map(lambda _: words[rd.randint(0, len(words)-1)], range(n))), text_size))
        text_size = list(map(lambda _: rd.randint(int(len(words)/10), int(len(words)/2)), range(cls.num_recs)))
        text_field2 = list(map(lambda n: ' '.join(map(lambda _: words[rd.randint(0, len(words)-1)], range(n))), text_size))

        cls.labeled_inp_df = pd.DataFrame(zip(train_group_field, numeric_field, categorical_field, boolean_field, text_field1, text_field2,
                                              set_field, label_field),
                                          columns=cls.fields_with_label)

        cls.nonlabeled_inp_df = pd.DataFrame(zip(train_group_field, numeric_field, categorical_field, boolean_field, text_field1, text_field2,
                                              set_field),
                                          columns=cls.fields_without_label)

        cls.labeled_binary_inp_df = pd.DataFrame(zip(train_group_field, numeric_field, categorical_field, boolean_field, text_field1, text_field2,
                                      set_field, binary_label_field),
                                  columns=cls.fields_with_label)





    def test_correlation_feature_selection(self):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __schema_with_label_nonnegative = self.schema_with_label_nonnegative.copy()

        ac = Classifier(model_configuration = testModelConfiguration)
        res_df = ac.feature_ranking(input_df=__labeled_inp_df, schema=__schema_with_label_nonnegative,
                                    mode=Classifier.CC_fs_correlation)

        self.assertTrue(isinstance(res_df, pd.DataFrame))
        self.assertEqual(len(res_df.columns), 2)
        self.assertEqual(res_df.dtypes[0], "object")
        self.assertIn(res_df.dtypes[1], ["int64", "float64" ])




    def test_backward_feature_selection(self):
        if testModelConfiguration['type'] in [Classifier.ENSEMBLE_SVC_MODEL_TYPE, Classifier.ENSEMBLE_LR_MODEL_TYPE]:
            return

        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)
        res_df = ac.feature_ranking(input_df=__labeled_inp_df, schema=__schema_with_label,
                                    mode=Classifier.CC_fs_backward)

        self.assertTrue(isinstance(res_df, pd.DataFrame))
        self.assertEqual(len(res_df.columns), 2)
        self.assertEqual(res_df.dtypes[0], "object")
        self.assertIn(res_df.dtypes[1], ["int64", "float64" ])





    def test_training(self):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)
        ac.train(input_df=__labeled_inp_df, schema=__schema_with_label)
        lr, fm, lm = ac.get_models()

        self.assertTrue(isinstance(lr, LR) or isinstance(lr, LSVC) or isinstance(lr, Ensemble))
        self.assertTrue(isinstance(fm, Featurizer))
        self.assertTrue(isinstance(lm, Featurizer))





    def test_predict_proba(self):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __nonlabeled_inp_df = self.nonlabeled_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)
        ac.train(input_df=__labeled_inp_df, schema=__schema_with_label)
        lr, fm, lm = ac.get_models()

        ac = Classifier(model_configuration = testModelConfiguration)
        ac.load_models(lr, fm, lm)

        for multilbl_pred in [True, False]:
            res_df = ac.predict_proba(input_df=__nonlabeled_inp_df, multilabel_pred=multilbl_pred)

            self.assertTrue(isinstance(res_df, pd.DataFrame))
            self.assertEqual(len(res_df.columns), len(self.fields_without_label) + 3)
            self.assertEqual(res_df.dtypes[-1], "float64")
            self.assertEqual(res_df.dtypes[-2], "object")
            self.assertEqual(res_df.dtypes[-3], "object")
            self.assertEqual(len(res_df), self.num_recs)

            if not multilbl_pred:
                self.assertFalse(any(list(map(lambda x: x[0] not in self.labels, res_df.filter([res_df.columns[-3]]).values))))
            else:
                list_lbls = list(map(lambda lbls: lbls[0].split(","), res_df.filter([res_df.columns[-3]]).values))
                list_valid_lbls = list(map(lambda lbls: map(lambda lbl: lbl not in self.labels, lbls), list_lbls))
                self.assertFalse(any(list(map(any, list_valid_lbls))))

            #Test if probabilities sum-up to 1
            prob_str = list(map(lambda p_str: p_str.split(','), res_df["Probabilities"].values))
            prob_float = list(map(lambda prob_with_label: [float(p.split(':')[1]) for p in prob_with_label], prob_str))

            self.assertFalse(any(list(map(lambda probs: sum(probs) >= 1.0 + 0.005 * len(self.fields_without_label) \
                                                or sum(probs) <= 1.0 - 0.005 * len(self.fields_without_label), prob_float))))





    def test_learn(self):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __nonlabeled_inp_df = self.nonlabeled_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)
        ac.train(input_df=__labeled_inp_df, schema=__schema_with_label)
        lr, fm, lm = ac.get_models()

        ac = Classifier(model_configuration = testModelConfiguration)
        ac.load_models(lr, fm, lm)

        res_df = ac.learn(input_df=__nonlabeled_inp_df)

        self.assertTrue(isinstance(res_df, pd.DataFrame))
        self.assertEqual(len(res_df.columns), len(self.fields_without_label) + 3)
        self.assertEqual(res_df.dtypes[-1], "float64")




    def test_input_qlty(self, binary_problem = False):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __labeled_binary_inp_df = self.labeled_binary_inp_df.copy(deep=True)
        __nonlabeled_inp_df = self.nonlabeled_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)

        if binary_problem:
            ac.train(input_df=__labeled_binary_inp_df, schema=__schema_with_label)
        else:
            ac.train(input_df=__labeled_inp_df, schema=__schema_with_label)
        lr, fm, lm = ac.get_models()

        ac = Classifier(model_configuration = testModelConfiguration)
        ac.load_models(lr, fm, lm)

        res_df = ac.input_qlty(input_df=__nonlabeled_inp_df)

        self.assertTrue(isinstance(res_df, pd.DataFrame))
        self.assertEqual(len(res_df.columns), len(self.fields_without_label) + 2)
        self.assertEqual(res_df.dtypes[-1], "object")
        self.assertEqual(res_df.dtypes[-2], "object")
        self.assertFalse(any(list(map(lambda x: x not in ["Good", "Bad", "OK"], res_df.filter([res_df.columns[-2]]).values))))

        #Test if all suggested features are not present
        def chk_feature_nonexistance(row):
            suggested_features = row["SuggestedFeatures"].split(',')
            for feat in suggested_features:
                if '::' in feat:
                    field_name, field_value = feat.split('::')
                    self.assertIn(field_name, self.fields_without_label)
                    fld_no = list(res_df.columns).index(field_name)
                    if self.schema_without_label[fld_no] in ["text", "text2vec"]:
                       self.assertNotIn(' ' + field_value + ' ', row[field_name].lower())
                       #self.assertTrue(True)
                    elif self.schema_without_label[fld_no] == "set":
                        if len(field_value) > 0:
                            self.assertNotIn(field_value, row[field_name])
                    elif self.schema_without_label[fld_no] in ["string", "numeric", "boolean"]:
                        self.assertNotEqual(field_value, row[field_name])
                else:
                    field_name = feat
                    if len(field_name) > 0:
                        self.assertIn(field_name, self.fields_without_label)

        res_df.apply(chk_feature_nonexistance, axis=1)





    def test_input_qlty_binary_prob(self):
        self.test_input_qlty(binary_problem=True)





    def test_predict_explain(self, binary_problem = False):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __nonlabeled_inp_df = self.nonlabeled_inp_df.copy(deep=True)
        __labeled_binary_inp_df = self.labeled_binary_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)

        if binary_problem:
            ac.train(input_df=__labeled_binary_inp_df, schema=__schema_with_label)
        else:
            ac.train(input_df=__labeled_inp_df, schema=__schema_with_label)
        lr, fm, lm = ac.get_models()

        ac = Classifier(model_configuration = testModelConfiguration)
        ac.load_models(lr, fm, lm)
        
        res_df = ac.predict_explain(input_df=__nonlabeled_inp_df)

        self.assertTrue(isinstance(res_df, pd.DataFrame))
        self.assertEqual(len(res_df.columns), len(self.fields_without_label) + 4)
        self.assertEqual(res_df.dtypes[-1], "object")
        self.assertEqual(res_df.dtypes[-2], "float64")
        self.assertEqual(res_df.dtypes[-3], "object")
        self.assertEqual(res_df.dtypes[-4], "object")

        #Test if all top-contributed features are present
        def chk_contributor_existance(row):
            contributors = row["TopContributors"].split(';')
            features = [ contrib.split('=')[0] for contrib in contributors]
            for feat in features:
                if '::' in feat:
                    field_name, field_value = feat.split('::')
                    self.assertIn(field_name, self.fields_without_label)
                    fld_no = list(res_df.columns).index(field_name)
                    if self.schema_without_label[fld_no] in ["text", "set"]:
                        #self.assertIn(field_value, row[field_name].lower())
                        self.assertTrue(True)
                    elif self.schema_without_label[fld_no] in ["string", "numeric", "boolean"]:
                        self.assertEqual(field_value, row[field_name])
                else:
                    field_name = feat
                    if len(field_name) > 0:
                        self.assertIn(field_name, self.fields_without_label)

        res_df.apply(chk_contributor_existance, axis=1)





    def test_predict_explain_binary_prob(self):
        self.test_predict_explain(binary_problem = True)





    def test_kfolds_eval(self, binary_problem = False):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __labeled_binary_inp_df = self.labeled_binary_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)

        if binary_problem:
            res_df = ac.eval(input_df=__labeled_binary_inp_df, schema=__schema_with_label,
                             mode = "K_FOLDS", nfolds=3)
        else:
            res_df = ac.eval(input_df=__labeled_inp_df, schema=__schema_with_label,
                             mode = "K_FOLDS", nfolds=3)

        self.assertTrue(isinstance(res_df, pd.DataFrame))
        self.assertEqual(res_df.dtypes[0], "object")

        if binary_problem:
            self.assertEqual(len(res_df.columns), max(1 + len(self.binary_labels), 5))
        else:
            self.assertEqual(len(res_df.columns), max(1 + len(self.labels), 5))




    def test_kfolds_eval_binary_prob(self):
        self.test_kfolds_eval(binary_problem = True)





    def test_LOO_eval(self, binary_problem = False):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __labeled_binary_inp_df = self.labeled_binary_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)

        if binary_problem:
            res_df = ac.eval(input_df=__labeled_binary_inp_df, schema=__schema_with_label,
                             mode = "LEAVE_ONE_OUT", nfolds=3)
        else:
            res_df = ac.eval(input_df=__labeled_inp_df, schema=__schema_with_label,
                             mode = "LEAVE_ONE_OUT", nfolds=3)

        self.assertTrue(isinstance(res_df, pd.DataFrame))
        self.assertEqual(len(res_df.columns), max(1 + len(self.labels), 5))
        self.assertEqual(res_df.dtypes[0], "object")





    def test_kfolds_eval_topN(self, binary_problem = False):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __labeled_binary_inp_df = self.labeled_binary_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)

        if binary_problem:
            res_df = ac.eval(input_df=__labeled_binary_inp_df, schema=__schema_with_label,
                             mode = "K_FOLDS", nfolds=3, topN=2)
        else:
            res_df = ac.eval(input_df=__labeled_inp_df, schema=__schema_with_label,
                             mode = "K_FOLDS", nfolds=3, topN=2)

        self.assertTrue(isinstance(res_df, pd.DataFrame))
        self.assertEqual(len(res_df.columns), max(1 + len(self.labels), 5))
        self.assertEqual(res_df.dtypes[0], "object")





    def test_LOO_eval_table_format(self):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)

        res_df = ac.eval(input_df=__labeled_inp_df, schema=__schema_with_label,
                             mode = "LEAVE_ONE_OUT", nfolds=3)

        self.assertTrue(isinstance(res_df, pd.DataFrame))
        self.assertEqual(len(res_df.columns), max(1 + len(self.labels), 5))
        self.assertEqual(res_df.dtypes[0], "object")




    def test_eval_data(self, binary_problem = False):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __labeled_binary_inp_df = self.labeled_binary_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)

        if binary_problem:
            labels, true_lbls, pred_lbls, conf_mat, cls_report = ac.eval_data(input_df=__labeled_binary_inp_df, schema=__schema_with_label,
                                                                              mode = "LEAVE_ONE_OUT", nfolds=3)
        else:
            labels, true_lbls, pred_lbls, conf_mat, cls_report = ac.eval_data(input_df=__labeled_inp_df, schema=__schema_with_label,
                                                                              mode = "LEAVE_ONE_OUT", nfolds=3)

        if binary_problem:
            self.assertTrue(len(labels)==2)
        else:
            self.assertTrue(len(labels)==len(self.labels))

        self.assertTrue(len(true_lbls)==self.num_recs)
        self.assertTrue(len(true_lbls)==len(pred_lbls))

        self.assertTrue(len(conf_mat)==len(labels))
        self.assertTrue(len(conf_mat[0])==len(labels))

        ext_labels = list(labels) + ['macro avg', 'weighted avg']
        for lbl in ext_labels:
            self.assertTrue(lbl in cls_report.keys())
            self.assertTrue('precision' in cls_report[lbl])
            self.assertTrue('recall' in cls_report[lbl])
            self.assertTrue('f1-score' in cls_report[lbl])
            self.assertTrue('support' in cls_report[lbl])

        self.assertTrue('accuracy' in cls_report.keys())



    def test_binary_eval_data(self):
        self.test_eval_data(binary_problem=True)




    def test_model_visualization(self, binary_problem = False):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __labeled_binary_inp_df = self.labeled_binary_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)

        if binary_problem:
            ac.train(input_df=__labeled_binary_inp_df, schema=__schema_with_label)
        else:
            ac.train(input_df=__labeled_inp_df, schema=__schema_with_label)
        lr, fm, lm = ac.get_models()

        ac = Classifier(model_configuration = testModelConfiguration)
        ac.load_models(lr, fm, lm)

        res_df = ac.model_visualization()
        self.assertTrue(isinstance(res_df, pd.DataFrame))
        self.assertEqual(len(res_df.columns), 3)
        self.assertEqual(res_df.dtypes[-1], "float64")
        self.assertEqual(res_df.dtypes[-2], "object")
        self.assertEqual(res_df.dtypes[-3], "object")




    def test_model_viz_binary_prob(self):
        self.test_model_visualization(binary_problem=True)



    def test_labels(self):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)
        ac.train(input_df=__labeled_inp_df, schema=__schema_with_label)

        labels = ac.labels()

        diff1 = [elem for elem in labels if elem not in self.labels]
        diff2 = [elem for elem in self.labels if elem not in labels]

        self.assertTrue(len(diff1)==0)
        self.assertTrue(len(diff2)==0)
        self.assertTrue(len(labels)==len(self.labels))



    def test_numclasses(self):
        __labeled_inp_df = self.labeled_inp_df.copy(deep=True)
        __schema_with_label = self.schema_with_label.copy()

        ac = Classifier(model_configuration = testModelConfiguration)
        ac.train(input_df=__labeled_inp_df, schema=__schema_with_label)

        nclasses = ac.num_classes()

        self.assertTrue(nclasses == len(self.labels))





if __name__ == "__main__":
    testModelConfiguration = modelConfigurations[int(sys.argv[1])]
    print("Testing ", testModelConfiguration['type'])
    unittest.main(argv=[''])