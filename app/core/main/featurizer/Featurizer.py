'''
Featurizer
'''
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import *

from app.core.main.featurizer.Doc2Vector import Doc2Vector




class Featurizer(object):

    FT_Numeric = "NUMERICAL"
    FT_String = "CATEGORICAL"
    FT_Text = "TEXT"
    FT_Set = "SET"
    FT_Boolean = "BOOLEAN"
    FT_Label = "LABEL"
    FT_TXT2V = "TEXT2VEC"
    FT_UNDEFINED = "UNDEFINED"

    FT_NOOP = "NOOP"
    FT_MIN_MAX_SCALER = "MIN_MAX_SCALER"
    FT_LABEL_BINARIZER = "LABEL_BINARIZER"
    FT_TFIDF_VECTORIZER = "TFIDF_VECTORIZER"
    FT_MULTILABEL_BINARIZER = "MULTILABEL_BINARIZER"
    FT_LABEL_ENCODER = "LABEL_ENCODER"
    FT_TEXT_TO_VECTOR = "TEXT_TO_VECTOR"


    supported_types = [FT_Numeric, FT_String, FT_Text, FT_TXT2V, FT_Set, FT_Boolean, FT_Label, FT_UNDEFINED,
                       FT_NOOP, FT_MIN_MAX_SCALER, FT_LABEL_BINARIZER, FT_TFIDF_VECTORIZER, FT_MULTILABEL_BINARIZER,
                       FT_LABEL_ENCODER, FT_TEXT_TO_VECTOR]



    def __init__(self, field_names, field_types, max_df, min_df, stop_words,
                 sublinear_tf, smooth_idf,
                 ngram_range, tokenizer):
        self.__field_names = field_names
        self.__field_types = field_types
        self.__models = []
        self.__models_data = []
        self.__featurizer_end_offset = []
        self.__tokenizer = tokenizer
        self.__max_df = max_df
        self.__min_df = min_df
        self.__stop_words = stop_words
        self.__sublinear_tf = sublinear_tf
        self.__smooth_idf = smooth_idf
        self.__ngram_range = ngram_range

        self.__feats = None



    def  __type_featurizer_map(self, type):
        assert type in Featurizer.supported_types, "Invalid type %s, supported types are %s" % (type, ','.join(Featurizer.supported_types))

        if type in [Featurizer.FT_Numeric, Featurizer.FT_MIN_MAX_SCALER]:
            return MinMaxScaler()
        elif type in [Featurizer.FT_Boolean, Featurizer.FT_String, Featurizer.FT_LABEL_BINARIZER]:
            return LabelBinarizer()
        elif type in [Featurizer.FT_Text, Featurizer.FT_TFIDF_VECTORIZER]:
            return TfidfVectorizer(input='content', max_df=self.__max_df, min_df=self.__min_df, stop_words=self.__stop_words,
                                decode_error='ignore', sublinear_tf=self.__sublinear_tf, smooth_idf=self.__smooth_idf,
                                ngram_range = self.__ngram_range, tokenizer = self.__tokenizer)
        elif type in [Featurizer.FT_Set, Featurizer.FT_MULTILABEL_BINARIZER]:
            return MultiLabelBinarizer()
        elif type in [Featurizer.FT_Label, Featurizer.FT_LABEL_ENCODER]:
            return LabelEncoder()
        elif type in [Featurizer.FT_TXT2V, Featurizer.FT_TEXT_TO_VECTOR]:
            return Doc2Vector()
        elif type in [Featurizer.FT_UNDEFINED, Featurizer.FT_NOOP]:
            return None



    def fit(self, data):
        '''
        :param data: must be a pandas DataFrame
        :return: None
        '''
        assert isinstance(data, pd.DataFrame), "Expect a DataFrame object"

        self.__models = []
        self.__models_data = []
        self.__featurizer_end_offset = []
        feat_offset = 0
        for (fieldNo, (fieldName, fieldData)) in enumerate(data.iteritems()):
            #debug
            print('Featurizer: fitting ' + fieldName)

            m = self.__type_featurizer_map(self.__field_types[fieldNo])

            if self.__field_types[fieldNo] in [Featurizer.FT_Numeric, Featurizer.FT_MIN_MAX_SCALER]:
                m.fit(fieldData.values.reshape(-1, 1))
                self.__models.append(m)
                feat_offset += 1
                self.__models_data.append(None)
            elif self.__field_types[fieldNo] in [Featurizer.FT_String, Featurizer.FT_LABEL_BINARIZER]:
                m.fit(fieldData.values)
                self.__models.append(m)
                feat_offset += len(self.__models[-1].classes_)
                self.__models_data.append(None)
            elif self.__field_types[fieldNo] in [Featurizer.FT_Text, Featurizer.FT_TFIDF_VECTORIZER]:
                m.fit(fieldData.values)
                self.__models.append(m)
                feat_offset += len(self.__models[-1].vocabulary_)
                self.__models_data.append(dict((fidx, w) for (w, fidx) in self.__models[-1].vocabulary_.items()))
            elif self.__field_types[fieldNo] in [Featurizer.FT_Set, Featurizer.FT_MULTILABEL_BINARIZER]:
                m.fit(fieldData.values)
                self.__models.append(m)
                feat_offset += len(self.__models[-1].classes_)
                self.__models_data.append(self.__models[-1].classes_)
            elif self.__field_types[fieldNo] in [Featurizer.FT_Boolean]:
                m.fit(fieldData.values)
                self.__models.append(m)
                feat_offset += 1
                self.__models_data.append(None)
            elif self.__field_types[fieldNo] in [Featurizer.FT_Label, Featurizer.FT_LABEL_ENCODER]:
                m.fit(fieldData.values)
                self.__models.append(m)
                feat_offset += 1
                self.__models_data.append(None)
            elif self.__field_types[fieldNo] in [Featurizer.FT_TXT2V, Featurizer.FT_TEXT_TO_VECTOR]:
                m.fit(fieldData.values)
                self.__models.append(m)
                feat_offset += m.vector_size()
                self.__models_data.append(None)
            elif self.__field_types[fieldNo] in [Featurizer.FT_UNDEFINED, Featurizer.FT_NOOP]:
                self.__models.append(m)
                feat_offset += 1
                self.__models_data.append(None)
            self.__featurizer_end_offset.append(feat_offset)



    def transform(self, data):
        '''
        :param data: must be a pandas DataFrame
        :return: numpy ndarray
        '''
        assert isinstance(data, pd.DataFrame), "Expect a DataFrame object"

        self.__feats = None
        this_col_feats = None
        for (fieldNo, (fieldName, fieldData)) in enumerate(data.iteritems()):
            #debug
            print('Featurizer: transforming ' + fieldName)

            m = self.__models[fieldNo]
            if self.__field_types[fieldNo] in [Featurizer.FT_Numeric, Featurizer.FT_MIN_MAX_SCALER]:
                this_col_feats = m.transform(fieldData.values.reshape(-1, 1))
            elif self.__field_types[fieldNo] in [Featurizer.FT_String, Featurizer.FT_LABEL_BINARIZER]:
                this_col_feats = m.transform(fieldData.values)
                #transform binary encoder to one-hot encoder in case there are 2 classes
                if len(m.classes_) == 2 and this_col_feats.shape[1] == 1:
                    this_col_feats = np.c_[1 - this_col_feats, this_col_feats]
            elif self.__field_types[fieldNo] in [Featurizer.FT_Text, Featurizer.FT_TFIDF_VECTORIZER]:
                this_col_feats = m.transform(fieldData.values)
            elif self.__field_types[fieldNo] in [Featurizer.FT_Set, Featurizer.FT_MULTILABEL_BINARIZER]:
                this_col_feats = m.transform(fieldData.values)
            elif self.__field_types[fieldNo] in [Featurizer.FT_Boolean]:
                this_col_feats = m.transform(fieldData.values)
            elif self.__field_types[fieldNo] in [Featurizer.FT_Label, Featurizer.FT_LABEL_ENCODER]:
                this_col_feats = m.transform(fieldData.values)
            elif self.__field_types[fieldNo] in [Featurizer.FT_TXT2V, Featurizer.FT_TEXT_TO_VECTOR]:
                this_col_feats = m.transform(fieldData.values)
            elif self.__field_types[fieldNo] in [Featurizer.FT_UNDEFINED, Featurizer.FT_NOOP]:
                this_col_feats = fieldData.values

            if issparse(this_col_feats):
                this_col_feats = this_col_feats.todense()
            if self.__feats is None:
                self.__feats = this_col_feats
            else:
                self.__feats = np.c_[self.__feats, this_col_feats]
        return self.__feats



    def inverse_transform(self, data):
        '''
        Use only for LABEL data.
        :param data: array like
        :return: array like
        '''
        assert len(self.__models)==1, "Expect to have exactly one model."
        lbls = self.__models[0].inverse_transform(data)
        return lbls



    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)



    def set_params(self, models, model_data, featurizer_offsets, tokenizer):
        self.__models = models
        self.__models_data = model_data
        self.__featurizer_end_offset = featurizer_offsets
        self.__tokenizer = tokenizer


    def get_params(self):
        return self.__models, self.__models_data, self.__featurizer_end_offset, self.__tokenizer


    def __remove_special_chars(self, inp_str):
        return inp_str.replace(':', ' ').replace('=', ' ').replace(',', ' ').replace(';', ' ')


    def get_feature(self, featNo):
        assert featNo < self.__featurizer_end_offset[-1], "Feature number %d is out of range!" % featNo

        begin_offset = 0
        for(fieldNo, m) in enumerate(self.__models):
            if self.__featurizer_end_offset[fieldNo] > featNo:
                feat_offset = featNo - begin_offset
                if self.__field_types[fieldNo] in [Featurizer.FT_Numeric, Featurizer.FT_MIN_MAX_SCALER]:
                    return self.__remove_special_chars(self.__field_names[fieldNo])
                elif self.__field_types[fieldNo] in [Featurizer.FT_String, Featurizer.FT_LABEL_BINARIZER]:
                    return self.__remove_special_chars(self.__field_names[fieldNo]) + "::" + \
                           self.__remove_special_chars(self.__models[fieldNo].classes_[feat_offset])
                elif self.__field_types[fieldNo] in [Featurizer.FT_Text, Featurizer.FT_TFIDF_VECTORIZER]:
                    return self.__remove_special_chars(self.__field_names[fieldNo]) + "::" + \
                           self.__remove_special_chars(self.__models_data[fieldNo][feat_offset])
                elif self.__field_types[fieldNo] in [Featurizer.FT_Set, Featurizer.FT_MULTILABEL_BINARIZER]:
                    return self.__remove_special_chars(self.__field_names[fieldNo]) + "::" + \
                           self.__remove_special_chars(self.__models_data[fieldNo][feat_offset])
                elif self.__field_types[fieldNo] in [Featurizer.FT_Boolean]:
                    return self.__remove_special_chars(self.__field_names[fieldNo])
                elif self.__field_types[fieldNo] in [Featurizer.FT_Label, Featurizer.FT_LABEL_ENCODER]:
                    return self.__remove_special_chars(self.__field_names[fieldNo])
                elif self.__field_types[fieldNo] in [Featurizer.FT_TXT2V, Featurizer.FT_TEXT_TO_VECTOR]:
                    return self.__remove_special_chars(self.__field_names[fieldNo]) + "::" + \
                           self.__remove_special_chars(str(feat_offset))
                elif self.__field_types[fieldNo] in [Featurizer.FT_UNDEFINED, Featurizer.FT_NOOP]:
                    return self.__remove_special_chars(self.__field_names[fieldNo])
            else:
                begin_offset = self.__featurizer_end_offset[fieldNo]



    def get_all_features(self):
        f_names = []
        for(fieldNo, m) in enumerate(self.__models):
            if self.__field_types[fieldNo] in [Featurizer.FT_Numeric, Featurizer.FT_MIN_MAX_SCALER]:
                f_names.append(self.__remove_special_chars(self.__field_names[fieldNo]))
            elif self.__field_types[fieldNo] in [Featurizer.FT_String, Featurizer.FT_LABEL_BINARIZER]:
                f_names += [self.__remove_special_chars(self.__field_names[fieldNo]) + "::" +
                            self.__remove_special_chars(cls) for cls in self.__models[fieldNo].classes_]
            elif self.__field_types[fieldNo] in [Featurizer.FT_Text, Featurizer.FT_TFIDF_VECTORIZER]:
                f_names += [self.__remove_special_chars(self.__field_names[fieldNo]) + "::" +
                            self.__remove_special_chars(word) for (widx, word) in \
                            sorted(self.__models_data[fieldNo].items(), key = lambda widx_w: widx_w[0])]
            elif self.__field_types[fieldNo] in [Featurizer.FT_Set, Featurizer.FT_MULTILABEL_BINARIZER]:
                f_names += [self.__remove_special_chars(self.__field_names[fieldNo]) + "::" +
                            self.__remove_special_chars(cls) for cls in self.__models_data[fieldNo]]
            elif self.__field_types[fieldNo] in [Featurizer.FT_Boolean]:
                f_names.append(self.__remove_special_chars(self.__field_names[fieldNo]))
            elif self.__field_types[fieldNo] in [Featurizer.FT_Label, Featurizer.FT_LABEL_ENCODER]:
                f_names.append(self.__remove_special_chars(self.__field_names[fieldNo]))
            elif self.__field_types[fieldNo] in [Featurizer.FT_TXT2V, Featurizer.FT_TEXT_TO_VECTOR]:
                f_names += [self.__remove_special_chars(self.__field_names[fieldNo]) + "::" +
                            self.__remove_special_chars(str(fidx)) for fidx in range(self.__models[fieldNo].vector_size())]
            elif self.__field_types[fieldNo] in [Featurizer.FT_UNDEFINED, Featurizer.FT_NOOP]:
                f_names.append(self.__remove_special_chars(self.__field_names[fieldNo]))
        return f_names



    def get_schema(self):
        return self.__field_types



    def __str__(self):
        return '''
        Featurizer
            Supported data types: %s
            For Tfidf vectorizer: 
                Min DF: %s
                Max DF: %s
                Sublinear TF: %s 
                Smooth IDF: %s
                Stop words: %s
                Tokenizer: %s
                ngrams range: %s
        ''' % (Featurizer.supported_types, self.__min_df, self.__max_df, self.__sublinear_tf, self.__smooth_idf,
               self.__stop_words, self.__tokenizer, self.__ngram_range)