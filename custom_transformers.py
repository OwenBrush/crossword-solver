
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from string import punctuation
from gensim.parsing.preprocessing import remove_stopwords

EXPRESSIONS_TO_REMOVE = ["\\"+x for x in list(punctuation)]
EXPRESSIONS_TO_REMOVE.remove('\\!')


class FeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(   self,
                    boolean_features=True,
                    clean_strings=True,
                    parse_stopwords=True,
                    count_strings=True,
                    min_characters:int = 1):
        
        self.boolean_features = boolean_features
        self.clean_strings = clean_strings
        self.parse_stopwords = parse_stopwords
        self.count_strings = count_strings
        self.min_characters = min_characters


    def fit(self, X, y = None):
        return self

    def transform(self, X:pd.DataFrame, y:pd.DataFrame = None):
        X = X.copy()
        if self.boolean_features: boolean_features(X)
        if self.clean_strings: clean_strings(X)
        if self.parse_stopwords:parse_stopwords(X)
        if self.count_strings :count_strings(X, self.min_characters)        
        if y is None:
            return X
        else:
            y = y.copy()
            if self.clean_strings: y = y.str.lower().str.strip()   
            return X, y
        
def boolean_features(X:pd.DataFrame):
    X['noun_involved'] = X['clue'].str.contains('[A-Z].*[A-Z]',regex=True).astype(float).astype('Int64')
    X['fill_blank'] = X['clue'].str.contains('_', regex=False).astype(float).astype('Int64')
    return X

def clean_strings(X:pd.DataFrame):
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
        
def count_strings(X:pd.DataFrame, min_characters_for_wordcount:int):
    X['word_count'] = X['clue'].str.count(f'\\b\\w{{{min_characters_for_wordcount},}}') 
    X['answer_length'] = X['answer_characters'].str.len()
    return X

def parse_stopwords(X:pd.DataFrame):
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