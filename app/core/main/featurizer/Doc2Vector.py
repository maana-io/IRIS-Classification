
import os
import numpy as np

from numpy.linalg import norm
from app.settings import W2VEC_MODEL_DIR



class Doc2Vector(object):
    #
    #    Pre-trained word2vec models of various languages with Creative Commons Attribution-Share-Alike License 3.0:
    #       https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
    #       https://fasttext.cc/docs/en/crawl-vectors.html
    #
    __W2V_file = W2VEC_MODEL_DIR + "/wiki-news-300d-100K.vec"
    __MAX_NUM_WORDS = 20000



    def __init__(self, model_file = None, max_num_words = -1):
        self.model_file_ = model_file if model_file is not None else self.__W2V_file
        self.max_num_words_ = max_num_words if max_num_words is not None else self.__MAX_NUM_WORDS

        assert os.access(self.model_file_, os.R_OK), "Failed to read from w2v model file %s" % self.model_file_
        self.__vectors = {}
        self.__vector_size = 0
        self.__vocab_size = 0




    #Load pre-trained model, no fitting
    def fit(self, data=None):
        if data is not None:
            inp_words = set('\n'.join(data).split())
        else:
            inp_words = None

        #fasttext format
        word_count = 0
        headerline = True
        with open(Doc2Vector.__W2V_file, 'rt') as txtFile:
            for line in txtFile:
                if headerline:
                    headerline = False
                    self.__vector_size = int(line.split()[1])
                else:
                    tokens = line.split()
                    if inp_words is None or tokens[0] in inp_words:
                        self.__vectors.update({tokens[0]: np.array([float(val) for val in tokens[1:]])})
                        word_count += 1
                        if word_count >= self.max_num_words_ and self.max_num_words_ >= 0:
                            break
        txtFile.close()
        self.__vocab_size = word_count



    #avg vector of word vectors
    def transform(self, data):
        feat_vectors = []
        for text in data:
            words = text.split()
            doc_vec = np.zeros(self.__vector_size, dtype=float)
            for w in words:
                if w in self.__vectors:
                    doc_vec += self.__vectors[w]
            if np.count_nonzero(doc_vec) > 0:
                doc_vec = doc_vec / norm(doc_vec)
            feat_vectors.append(doc_vec)
        return feat_vectors



    def vector_size(self):
        return self.__vector_size



    def vocab_size(self):
        return self.__vocab_size



    def __str__(self):
        return '''
        Pre-trained word embedding model: %s
        Vector size: %d
        Vocabulary size: %d
        ''' % (Doc2Vector.__W2V_file, self.vector_size(), self.vocab_size())




