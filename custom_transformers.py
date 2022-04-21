
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from string import punctuation
from gensim.parsing.preprocessing import remove_stopwords

EXPRESSIONS_TO_REMOVE = ["\\"+x for x in list(punctuation)]
EXPRESSIONS_TO_REMOVE.remove('\\!')


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, models:list = None, min_characters_for_wordcount:int = 1, remove_stop_words=True):
        self.min_characters_for_wordcount = min_characters_for_wordcount
        self.remove_stop_words = remove_stop_words

    def fit(self, X, y = None):
        return self

    def transform(self, X:pd.DataFrame, y:pd.DataFrame = None):
        X = X.copy()
        BooleanFeatures(X)
        CleanStrings(X)
        if self.remove_stop_words:
            ParseStopwords(X)
        CountStrings(X, self.min_characters_for_wordcount)
        
        if y is None:
            return X
        else:
            y = y.copy()
            y = y.str.lower().str.strip()   
            return X, y
        
def BooleanFeatures(X:pd.DataFrame):
    X['noun_involved'] = X['clue'].str.contains('[A-Z].*[A-Z]',regex=True)
    X['fill_blank'] = X['clue'].str.contains('_', regex=False)
    return X

def CleanStrings(X:pd.DataFrame):
    X['clue'] = X['clue'].str.lower().str.strip()
    X['clue'] = X['clue'].replace('$', ' money ', regex=False)
    X['clue'] = X['clue'].replace('!', ' ! ', regex=False)
    # X['clue'] = X['clue'].replace('``', '"', regex=False)
    X['clue'] = X['clue'].replace(r'\b\w{1,1}\b','', regex=True) 
    X['clue'] = X['clue'].replace(EXPRESSIONS_TO_REMOVE, ' ',regex=True)
    X['clue'] = X['clue'].replace('\d+', '', regex=True)
    X['clue'] = X['clue'].replace(' +', ' ', regex=True)
    X['clue'] = X['clue'].replace(['nan',''], np.nan, regex=False)
    return X
        
def CountStrings(X:pd.DataFrame, min_characters_for_wordcount:int):
    X['word_count'] = X['clue'].str.count(f'\\b\\w{{{min_characters_for_wordcount},}}') 
    X['answer_length'] = X['answer_characters'].str.len()
    return X

def ParseStopwords(X:pd.DataFrame):
    clues_without_stops = X['clue'].astype(str).apply(remove_stopwords)
    filter = clues_without_stops != ''
    X.loc[filter, 'clue'] = clues_without_stops[filter] 
    return X
    
class CosineSimilarity(BaseEstimator, TransformerMixin):
    
    def __init__(self, gensim_models:list):
        self.gensim_models = gensim_models
        
    def set_word2vec_models(self, gensim_models:list):
        self.gensim_models = gensim_models
    
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        for model in self.gensim_models:
           pass
        if y is None:
            return X
        else: 
            return X, y