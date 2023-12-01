import openai
import pandas as pd
import os
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer



class SpamClassifier:
    def __init__(self, api_key):
        self.__vectorizer = CountVectorizer(stop_words='english')
        self.__api_key = "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"
        self.__client  = openai.OpenAI(api_key=self.__api_key)
        self.__remove_non_alphabets =lambda x: re.sub(r'[^a-zA-Z]',' ',x)

    def __clean_data(self, df, column):
        print('Processing : [=', end='')
        df[column] = df[column].apply(self.__remove_non_alphabets)
        print('=', end='')
        df[column] = df[column].apply(lambda x: ' '.join(x))
        print('] : Completed', end='')

        self.__remove_stop_words(df=df, column=column)


    def __remove_stop_words(self, df, column):
        print("data frame: ",df)
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df[column])
        feature_names = vectorizer.get_feature_names_out()
        df[column] = feature_names[0]

        return df
    
    def classify_as_spam(self, message):

        try :
            response = self.__client.chat.completions.create(
                model="text-davinci-002",
                messages=message,
                max_tokens=50,
                temperature=0.7
            )
            generated_label = response.choices[0].text.strip().lower()

            if "spam" in generated_label:
                return "Yes"
            else:
                return "No"

        except Exception as e:
            print(e)
            return "No"
        
    def process_data(self, message):
        result = self.classify_as_spam(message)
        return result



        


def clean_data(message):
    vectorizer = CountVectorizer(stop_words='english')

    df = pd.DataFrame({"text": [message]})
    print(df)
    X = vectorizer.fit_transform(df['text'])

    feature_names = vectorizer.get_feature_names_out()
    dense_array = X.toarray()

    return feature_names[0]
