'''
WordNet Lemmatizer
'''
from nltk.stem import WordNetLemmatizer
from app.core.main.tokenizer.BaseTokenizer import BaseTokenizer

class LemmaTokenizer(object):
    def __init__(self):
        self.__wnl = WordNetLemmatizer()
        self.__basetokenizer = BaseTokenizer()


    def __call__(self, doc):
        return self.tokenize(doc)


    def tokenize(self, doc):
        return [self.__wnl.lemmatize(t) for t in self.__basetokenizer.tokenize(doc)]


    def __str__(self):
        return '''
        WordNet Lemmatizer based on
        %s
        ''' % self.__basetokenizer