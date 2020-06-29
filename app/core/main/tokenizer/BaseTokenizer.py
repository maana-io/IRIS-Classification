'''
Base Tokenizer
'''

from nltk import word_tokenize


class BaseTokenizer(object):

    def __init__(self):
        pass

    def __call__(self, doc):
        return self.tokenize(doc)

    def tokenize(self, doc):
        tokens = [t for t in word_tokenize(doc)]
        return tokens


    def __str__(self):
        return 'Base tokenizer.'