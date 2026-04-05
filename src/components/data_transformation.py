import os,sys

from src.constants import *
from src.utils.main_utils import save_object
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag


from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.over_sampling import SMOTE




class DataTransformation:
    def __init__(self):
        pass

    @staticmethod
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return 'a'
        elif tag.startswith('V'):
            return 'v'
        elif tag.startswith('N'):
            return 'n'
        elif tag.startswith('R'):
            return 'r'
        else:
            return 'n'

    @staticmethod   
    def preprocess_text(text):
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        tokens = word_tokenize(text)

        tokens = [w for w in tokens if w not in stop_words]

        pos_tags = pos_tag(tokens)

        tokens = [
            lemmatizer.lemmatize(word,DataTransformation.get_wordnet_pos(pos))
            for word, pos in pos_tags
        ]

        return " ".join(tokens)
    
    def initiate_data_transformation(self,train_file_path:str,test_file_path:str):
        try:
            logging.info('Data Transformation Started') 
            train_df=pd.read_csv(train_file_path)
            test_df=pd.read_csv(test_file_path)
            
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

            train_df.drop_duplicates(inplace=True)
            test_df.drop_duplicates(inplace=True)

            train_df['text']=train_df['text'].apply(DataTransformation.preprocess_text)
            test_df['text']=test_df['text'].apply(DataTransformation.preprocess_text)


            x_train_df=train_df['text']
            y_train_df=train_df['text_type']

            x_test_df=test_df['text']
            y_test_df=test_df['text_type']

            target_encoder=LabelEncoder()
            y_train_arr=target_encoder.fit_transform(y_train_df)
            y_test_arr=target_encoder.transform(y_test_df)

            #saving target label encoder
            save_object('artifacts/target_encoder.pkl',target_encoder)

            vectorizer = TfidfVectorizer(max_features=5000)
            x_train_arr = vectorizer.fit_transform(x_train_df)
            x_test_arr = vectorizer.transform(x_test_df)

            #saving tfidfvectorizer
            save_object('artifacts/tfidf_vectorizer.pkl',vectorizer)


            smote = SMOTE(random_state=42)

            x_train_resampled, y_train_resampled = smote.fit_resample(x_train_arr, y_train_arr)

            # train_arr=np.c_[x_train_resampled, y_train_resampled]
            # test_arr=np.c_[x_test_arr,y_test_arr]


            return x_train_resampled, y_train_resampled,x_test_arr,y_test_arr

        except Exception as e :
            raise CustomException(e,sys)
        
                

    
        
