
import numpy as np
import pandas as pd
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from string import punctuation
from gensim.parsing.preprocessing import remove_stopwords

EXPRESSIONS_TO_REMOVE = ["\\"+x for x in list(punctuation)]
EXPRESSIONS_TO_REMOVE.remove('\\!')
EXPRESSIONS_TO_REMOVE.remove('\\$')

class StringFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, min_characters_for_wordcount:int = 1, percent_of_known_characters:float=0):
        self.min_characters_for_wordcount = min_characters_for_wordcount
        self.percent_of_known_characters = percent_of_known_characters

    def fit(self, X, y = None):    
        return self
    
    def transform(self, X:pd.DataFrame, y:pd.Series = None):
        X = X.copy()
        self.boolean_features(X)
        self.clean_strings(X)
        self.parse_stopwords(X)
        self.count_clue_words(X)        
        if y is None:
            return X
        else:
            y = y.copy()
            y = y.str.lower().str.strip()   
            self.answer_lengths(X, y)
            self.known_characters(X, y)
            return X, y    
        
    def boolean_features(self, X:pd.DataFrame):
        X['noun_involved'] = X['clue'].str.contains('[A-Z].*[A-Z]',regex=True).astype(float).astype('Int64')
        X['fill_blank'] = X['clue'].str.contains('_', regex=False).astype(float).astype('Int64')
        return X

    def clean_strings(self, X:pd.DataFrame):
        X['clue'] = X['clue'].str.lower().str.strip()
        X['clue'] = X['clue'].replace('$', ' $ ', regex=False)
        X['clue'] = X['clue'].replace('!', ' ! ', regex=False)
        X['clue'] = X['clue'].replace(r'\b\w{1,1}\b','', regex=True) 
        X['clue'] = X['clue'].replace(EXPRESSIONS_TO_REMOVE, ' ',regex=True)
        X['clue'] = X['clue'].replace('\d+', '', regex=True)
        X['clue'] = X['clue'].replace(' +', ' ', regex=True)
        X['clue'] = X['clue'].replace(['nan',''], np.nan, regex=False)
        return X
            
    def parse_stopwords(self, X:pd.DataFrame):
        clues_without_stops = X['clue'].astype(str).apply(remove_stopwords)
        filter = clues_without_stops != ''
        X.loc[filter, 'clue'] = clues_without_stops[filter] 
        return X
    
    def count_clue_words(self, X:pd.DataFrame):
        X['word_count'] = X['clue'].str.count(f'\\b\\w{{{self.min_characters_for_wordcount},}}') 
        # X['answer_length'] = X['answer_characters'].str.len()
        return X

    def answer_lengths(self, X:pd.DataFrame, y:pd.Series):
        X['answer_length'] = y.str.len().astype(int)
        return X
        
    def known_characters(self, X:pd.DataFrame, y:pd.Series):
        def random_character_assignment(text, percent):
            known_characters = random.sample(range(len(text)),round(len(text)*percent))
            new_text = ''
            for i, x in enumerate(text):
                if i in known_characters:
                    new_text+=x
                else:
                    new_text+='_'
            return new_text
        X['answer_characters'] = y.apply(random_character_assignment, args=[self.percent_of_known_characters])
        return X


class PCAFeatures(BaseEstimator, TransformerMixin):
    def __init__(   self, model_dict:dict, pca_components:int = 10):
        
        self.model_dict = model_dict
        self.pca_components = pca_components


    def fit(self, X, y = None):
        self.word_vector_dict = {}
        self.pca_dict = {}
        for model_name, model in self.model_dict.items():
            #Convert clues to vectors
            vocab = model.index_to_key
            clues = X['clue'].astype(str).apply(lambda clue: [x for x in clue.split() if x in vocab])
            filter = clues.str.len() > 0
            clue_vectors = np.array([np.mean(model[x],axis=0) for x in clues[filter]]) 
            self.word_vector_dict[model_name] = clue_vectors
            #Train PCA
            pca = PCA(n_components=self.pca_components)   
            pca.fit(clue_vectors)
            self.pca_dict = pca       
        return self

    def transform(self, X:pd.DataFrame, y:pd.DataFrame = None):
        X = X.copy()
        self.apply_pca(X)
        if y is None:
            return X
        else:
            return X, y
        
    def apply_pca(self, X:pd.DataFrame):
        X = X.copy()
        for model_name, model in self.model_dict.items():
            #Convert clues to vectors
            vocab = model.index_to_key
            clues = X['clue'].astype(str).apply(lambda clue: [x for x in clue.split() if x in vocab])
            clues = clues[clues.str.len() > 0]
            clue_vectors = np.array([np.mean(model[x],axis=0) for x in clues])
            
            pca = pd.DataFrame(self.pca_dict[model_name].transform(clue_vectors))
            pca.index = clues.index
            pca.columns = [f'{model_name}_{x}' for x in pca.columns]
            X = pd.concat([X,pca],axis=1).fillna(0)
        return X
    