
from unittest import mock
import numpy as np
import pandas as pd
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, median_absolute_error
from string import punctuation
from gensim.parsing.preprocessing import remove_stopwords

EXPRESSIONS_TO_REMOVE = ["\\"+x for x in list(punctuation)]
EXPRESSIONS_TO_REMOVE.remove('\\!')
EXPRESSIONS_TO_REMOVE.remove('\\$')

CHARACTER_PERCENTAGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
COSINE_PREDICTION_FEATURES = ['noun_involved', 
                              'fill_blank', 
                              'word_count', 
                              'answer_length',                   
                              'twitter_0', 
                              'twitter_1', 
                              'twitter_2', 
                              'twitter_3', 
                              'twitter_4',
                              'twitter_5', 
                              'twitter_6', 
                              'twitter_7', 
                              'twitter_8', 
                              'twitter_9',
                              'google_0', 
                              'google_1', 
                              'google_2', 
                              'google_3', 
                              'google_4', 
                              'google_5',
                              'google_6', 
                              'google_7', 
                              'google_8', 
                              'google_9', 
                              'wiki_0', 
                              'wiki_1',
                              'wiki_2', 
                              'wiki_3', 
                              'wiki_4', 
                              'wiki_5', 
                              'wiki_6', 
                              'wiki_7', 
                              'wiki_8',
                              'wiki_9']

class StringFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, min_characters_for_wordcount:int = 1, percents_of_known_characters:list=CHARACTER_PERCENTAGES):
        self.min_characters_for_wordcount = min_characters_for_wordcount
        self.percents_of_known_characters = percents_of_known_characters

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
        for percent in self.percents_of_known_characters:
            X[f'{percent*100}%_known_characters'] = y.apply(random_character_assignment, args=[percent])
        return X


class PCAFeatures(BaseEstimator, TransformerMixin):
    def __init__(   self, gensim_model_dict:dict, pca_components:int = 10):
        
        self.gensim_model_dict = gensim_model_dict
        self.pca_components = pca_components


    def fit(self, X, y = None):
        self.word_vector_dict = {}
        self.pca_dict = {}
        for model_name, model in self.gensim_model_dict.items():
            #Convert clues to vectors
            vocab = model.index_to_key
            clues = X['clue'].astype(str).apply(lambda clue: [x for x in clue.split() if x in vocab])
            filter = clues.str.len() > 0
            clue_vectors = np.array([np.mean(model[x],axis=0) for x in clues[filter]]) 
            self.word_vector_dict[model_name] = clue_vectors
            #Train PCA
            pca = PCA(n_components=self.pca_components)   
            pca.fit(clue_vectors)
            self.pca_dict[model_name] = pca       
        return self

    def transform(self, X:pd.DataFrame, y:pd.Series = None):
        X = X.copy()
        self.apply_pca(X)
        if y is None:
            return X
        else:
            return X, y
        
    def apply_pca(self, X:pd.DataFrame):
        for model_name, model in self.gensim_model_dict.items():
            #Convert clues to vectors
            vocab = model.index_to_key
            clues = X['clue'].astype(str).apply(lambda clue: [x for x in clue.split() if x in vocab])
            filter = clues.str.len() > 0
            clues = clues[filter]
            clue_vectors = np.array([np.mean(model[x],axis=0) for x in clues])
            
            pca_features = pd.DataFrame(self.pca_dict[model_name].transform(clue_vectors))
            pca_features.index = clues.index
            pca_features.columns = [f'{model_name}_{x}' for x in pca_features.columns]
            for feature in pca_features.columns:
                X.loc[filter, feature] = pca_features[feature]
                X[feature].fillna(0,inplace=True)
        return X
    
    
class SimilarityPrediction(BaseEstimator, TransformerMixin):
    def __init__(   self, gensim_model_dict:dict, predictor_dict:dict,):
        
        self.gensim_model_dict = gensim_model_dict
        self.predictor_dict = predictor_dict
        self.predictions = {}


    def fit(self, X:pd.DataFrame, y:pd.Series = None):
        features = X[COSINE_PREDICTION_FEATURES] 
        for model_name, model in self.gensim_model_dict.items():
            filter = X[f'{model_name}_cosine_similarity'].notna()
            cosines = X[filter][f'{model_name}_cosine_similarity']
            predictor = self.predictor_dict[model_name]
            predictor.fit(features[filter], cosines)
        pass

    def transform(self, X:pd.DataFrame, y:pd.Series = None):
        X = X.copy()
        features = X[COSINE_PREDICTION_FEATURES] 
        for model_name, model in self.gensim_model_dict.items():
            predictor = self.predictor_dict[model_name]
            cosines = predictor.predict(features)
            X[f'{model_name}_predicted_similarity'] = cosines
            filter = X[f'{model_name}_cosine_similarity'].notna()
            
            true = X[filter][f'{model_name}_cosine_similarity']
            predict = X[filter][f'{model_name}_predicted_similarity']
            mean_error = mean_absolute_error(true,predict)
            median_error = median_absolute_error(true,predict)
            print(f'{model_name}, {predictor}\nMean Absolute Error: {mean_error}\nMedian Absolute Error: {median_error}')
        if y is None:
            return X
        else:
            return X, y
        
