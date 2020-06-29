'''
Porter Tokenizer
'''
from nltk.stem.porter import PorterStemmer
from app.core.main.tokenizer.BaseTokenizer import BaseTokenizer


class PorterTokenizer(object):

    def __init__(self):
        self.__wnl = PorterStemmer()
        self.__basetokenizer = BaseTokenizer()


    def __call__(self, doc):
        return self.tokenize(doc)


    def tokenize(self, doc):
        return [self.__wnl.stem(t) for t in self.__basetokenizer.tokenize(doc)]


    def __str__(self):
        return '''
        Porter tokenizer based on 
        %s
        ''' % self.__basetokenizer