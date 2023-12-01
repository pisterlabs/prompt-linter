import pandas as pd
import numpy as np
import joblib
import timeit
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocess import preprocess
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

class Asp_Sentiment_classifiers:
    """
    Loads all models provides functions to access them.
    Receives dataframe and performs predictions on them.
    """
    def __init__(self):
        '''
        Loads all models on startup once.
        '''
        # Load aspect model

        # lda = gensim.models.LdaModel.load(' models/lda.model')

        # Load sentiment models
        
        # Staff
        # json_file = open("models/staff_sentiment_model.json", 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # staff_model = model_from_json(loaded_model_json)
        # staff_model.load_weights("models/staff_sentiment_model.h5")

        # # Amenities
        # json_file = open("models/amenities_sentiment_model.json", 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # amenities_model = model_from_json(loaded_model_json)
        # amenities_model.load_weights("models/amenities_sentiment_model.h5")

        # # Condition
        # json_file = open("models/condition_sentiment_model.json", 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # condition_model = model_from_json(loaded_model_json)
        # condition_model.load_weights("models/condition_sentiment_model.h5")

        # # Cleanliness
        # json_file = open("models/cleanliness_sentiment_model.json", 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # cleanliness_model = model_from_json(loaded_model_json)
        # cleanliness_model.load_weights("models/cleanliness_sentiment_model.h5")

        # Overall
        json_file = open("models/sentiment_model2.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        overall_model = model_from_json(loaded_model_json)
        overall_model.load_weights("models/sentiment_model2.h5")

        # Aspect
        # lda_model = gensim.models.LdaModel.load('models/lda.model')
        json_file = open("models/aspect_model2.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        aspect_model = model_from_json(loaded_model_json)
        aspect_model.load_weights("models/aspect_model2.h5")

        tokenizer = Tokenizer(num_words=5000)
        tokenizer = joblib.load("models/tokenizer2.pkl")

        # self.staff_model = staff_model
        # self.amenities_model = amenities_model
        # self.condition_model = condition_model
        # self.cleanliness_model = cleanliness_model
        self.overall_model = overall_model
        self.aspect_model = aspect_model
        # self.aspect_model = lda_model
        self.tokenizer = tokenizer

    # def get_staff_model(self):
    #     return self.staff_model

    # def get_amenities_model(self):
    #     return self.amenities_model
    
    # def get_condition_model(self):
    #     return self.amenities_model
    
    # def get_cleanliness_model(self):
    #     return self.cleanliness_model

    # def get_overall_model(self):
    #     return self.overall_model

    # def get_aspect_model(self):
    #     return self.aspect_model

    # def aspects(self, df, corpus):
    #     """
    #     Classifies each sentence by aspect(s).
    #     Returns dataframe with additional columns for each aspect.
    #     """
    #     lda_model = self.aspect_model
    #     corpus = corpus
    #     def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=df['clean_text'].tolist(), review_id=df['review_id']):
    #         # Init output
    #         sent_topics_df = pd.DataFrame()

    #         # Get main topic in each document
    #         for i, row in enumerate(ldamodel[corpus]):
    #             row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
    #             # row = sorted(row, key=lambda x: (x[1]), reverse=True) # old line
    #             # Get the Dominant topic, Perc Contribution and Keywords for each document
    #             for j, (topic_num, prop_topic) in enumerate(row):
    #                 if j == 0: # => dominant topic
    #                     wp = ldamodel.show_topic(topic_num)
    #                     topic_keywords = ", ".join([word for word, prop in wp])
    #                     sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
    #                 else:
    #                     break
    #         sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    #         # Add original text to the end of the output
    #         review = pd.Series(review_id)
    #         contents = pd.Series(texts)
    #         sent_topics_df = pd.concat([review, sent_topics_df, contents], axis=1)
    #         return(sent_topics_df)

    #     df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=df['clean_text'].tolist(), review_id=df['review_id'])

    #     # Format
    #     df_dominant_topic = df_topic_sents_keywords.reset_index()
    #     df_dominant_topic.columns = ['Sentence_No', 'review_id', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    #     def classify(df_dominant_topic, df):
    #         output_df = df
    #         pass

    #     return output_df

    def sentiments(self, reviews_df):
        """
        After classifying aspects, classify overall sentiments and then sentiments per aspect for all reviews/sentences.
        Returns dataframe
        """

        tokenizer = self.tokenizer
        # Vectorize/tokenize sentence with vocabulary
        maxlen=100
        X_data = tokenizer.texts_to_sequences(reviews_df['stem_text'])
        X_data = pad_sequences(X_data, padding='post', maxlen=maxlen)

        # Overall Sentiment Predictions
        # y_pred = self.overall_model.predict(X_data, batch_size=5, verbose=1)
        # y_pred_bool = np.argmax(y_pred, axis=1)
        
        # predictions = []
        # for pred in y_pred_bool:
        #     if pred == 0:
        #         predictions.append('positive')
        #     elif pred == 1:
        #         predictions.append('neutral')
        #     else:
        #         predictions.append('negative')
        # reviews_df['overall_sentiment'] = predictions


        # Sentiments per Aspect

        def get_labels(y_pred):     
            if y_pred == 0:
                return 'positive'
            elif y_pred == 1:
                return 'neutral'
            else:
                return 'negative'

        reviews_df['overall_sentiment'] = None
        reviews_df['staff_sent'] = None
        reviews_df['amenities_sent'] = None
        reviews_df['condition_sent'] = None
        reviews_df['clean_sent'] = None
        print('Running... Please wait...')
        for i, aspect in zip(range(len(X_data)), reviews_df['aspect']):
            y_pred = -1
            if "a1" in aspect:
                y_pred = self.overall_model.predict(X_data[i:i+1], batch_size=5, verbose=0)
                y_pred_bool = np.argmax(y_pred, axis=1)
                reviews_df['staff_sent'][i] = get_labels(y_pred_bool)
                reviews_df['overall_sentiment'][i] = get_labels(y_pred_bool) 
            elif "a2" in aspect:
                y_pred = self.overall_model.predict(X_data[i:i+1], batch_size=5, verbose=0)
                y_pred_bool = np.argmax(y_pred, axis=1)
                reviews_df['amenities_sent'][i] = get_labels(y_pred_bool) 
                reviews_df['overall_sentiment'][i] = get_labels(y_pred_bool) 
            elif "a3" in aspect:
                y_pred = self.overall_model.predict(X_data[i:i+1], batch_size=5, verbose=0)
                y_pred_bool = np.argmax(y_pred, axis=1)
                reviews_df['condition_sent'][i] = get_labels(y_pred_bool) 
                reviews_df['overall_sentiment'][i] = get_labels(y_pred_bool) 
            elif "a4" in aspect:
                y_pred = self.overall_model.predict(X_data[i:i+1], batch_size=5, verbose=0)
                y_pred_bool = np.argmax(y_pred, axis=1)
                reviews_df['clean_sent'][i] = get_labels(y_pred_bool) 
                reviews_df['overall_sentiment'][i] = get_labels(y_pred_bool) 
            else:
                y_pred = self.overall_model.predict(X_data[i:i+1], batch_size=5, verbose=1)
                y_pred_bool = np.argmax(y_pred, axis=1)
                reviews_df['overall_sentiment'][i] = get_labels(y_pred_bool) 
            
            # else:
            #     y_pred = self.overall_model.predict(X_data, batch_size=5, verbose=1)
           
        

        return reviews_df

    def aspect_classifier(self, reviews_df, threshold=0.3):
        """
        Classifies according to aspect.
        Returns dataframe with aspect per sentence.
        """
        tokenizer = self.tokenizer
        # Vectorize/tokenize sentence with vocabulary
        maxlen=100
        X_data = tokenizer.texts_to_sequences(reviews_df['stem_text'])
        X_data = pad_sequences(X_data, padding='post', maxlen=maxlen)

        y_pred = self.aspect_model.predict(X_data, batch_size=5, verbose=1)
        y_pred_bool = y_pred
        # Threshold *requires tuning*
        y_pred_bool[y_pred_bool>=threshold] = 1
        y_pred_bool[y_pred_bool<threshold] = 0
        
        predictions = [[] for _ in range(len(y_pred_bool))]

        for index, values in zip(range(len(y_pred_bool)), y_pred_bool):
            if values[0] == 1:
                predictions[index].append('a1')
            if values[1] == 1:
                predictions[index].append('a2')
            if values[2] == 1:
                predictions[index].append('a3')
            if values[3] == 1:
                predictions[index].append('a4')

        reviews_df['aspect'] = predictions

        return reviews_df