import numpy as np
import pandas as pd
import random
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from string import punctuation
from gensim.parsing.preprocessing import remove_stopwords
from tqdm import tqdm


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
        for model_name, model in tqdm(self.gensim_model_dict.items()):
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
        for model_name, model in tqdm(self.gensim_model_dict.items()):
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
        for model_name, model in tqdm(self.gensim_model_dict.items()):
            filter = X[f'{model_name}_cosine_similarity'].notna()
            cosines = X[filter][f'{model_name}_cosine_similarity']
            predictor = self.predictor_dict[model_name]
            predictor.fit(features[filter], cosines)
        pass

    def transform(self, X:pd.DataFrame, y:pd.Series = None):
        X = X.copy()
        features = X[COSINE_PREDICTION_FEATURES] 
        for model_name, model in tqdm(self.gensim_model_dict.items()):
            predictor = self.predictor_dict[model_name]
            cosines = predictor.predict(features)
            X[f'{model_name}_predicted_similarity'] = cosines
        if y is None:
            return X
        else:
            return X, y
        

class SelectTopNWords():
    def __init__(self, topN:int=5):   
        self.topN = topN
        self.all_predictions ={}
        
    def predict(self, X:pd.DataFrame, known_characters:pd.Series, gensim_models:dict):
        self.all_predictions ={}
        for model_name, model in gensim_models.items():
            model_predictions = {}
            word_vectors = self.vectorize_sentences(X['clue'], model)
            for index, vector in word_vectors.iteritems():
                target = X[f'{model_name}_predicted_similarity'].loc[index]
                regex_pattern = re.compile('^'+''.join([x if not x == '_' else '[a-z]' for x in known_characters[index]])+'$')
                similarity_index = model.similar_by_vector(vector, topn=len(model.index_to_key))
                available_words = [x[0] for x in similarity_index if regex_pattern.match(x[0]) ]
                similarity_scores = np.asarray([x[1] for x in similarity_index if regex_pattern.match(x[0]) ])
                chosen_indices = np.abs(similarity_scores - target ).argsort()[:self.topN*2]
                word_matches = {}
                for i in chosen_indices:
                    word_matches[available_words[i]] = 1 - abs(target - similarity_scores[i])
                model_predictions[index] = word_matches
            self.all_predictions[model_name] = model_predictions
        
        return self.compile_Predictions(self.all_predictions)
    
    def compile_Predictions(self, predictions):
        final_words = {}
        final_scores = {}
        for i, row in pd.DataFrame(predictions).iterrows():
            votes = {}
            for chosen_words in row:
                if not chosen_words is np.nan:
                    for word, score in chosen_words.items():
                        if word in votes:
                            votes[word]+=score
                        else:
                            votes[word]= score
            votes = sorted(votes.items(), key= lambda kv: kv[1], reverse=True)[:self.topN]
            final_words[i] = [vote[0] for vote in votes]
            final_scores[i] = [vote[1]/len(predictions) for vote in votes]          
        return pd.DataFrame(final_words).T, pd.DataFrame(final_scores).T    
    

    def vectorize_sentences(self, strings:pd.Series, model):
        vocab = model.index_to_key
        clues = strings.astype(str).apply(lambda clue: [x for x in clue.split() if x in vocab])
        df_filter = clues.str.len() > 0
        clues = clues[df_filter]
        clue_vectors = pd.Series([np.mean(model[x],axis=0) for x in clues])
        clue_vectors.index = clues.index
        return clue_vectors