import pandas as pd
import json
import os
import time
from cohere_classifier.cohere_classifier import *
from openai_classifier.openai_classifier import *

if __name__ == "__main__":
    with open("../api-keys.json", "r") as f:
        api_keys = json.load(f)
        print(f"Found keys for {', '.join(api_keys.keys())}")

    #-----------------------------Cohere Classifier------------------------------------------------

    # CH_Classifier = CohereClassifier(api_keys)
    # input = ["""the economic gets better, people buy the stocks"""]
    # rate = CH_Classifier.get_ratings(input)
    # print(rate)

    #-----------------------------Openai davinci Classifier----------------------------------------
    # text_reviews = [
    #     "The product was amazing! Absolutely loved it.",
    #     "It's just okay, not great, but not terrible either.",
    #     "The worst experience I've ever had with a product. Terrible!"
    # ]
    # Openai_classifier = OpenaiClassifier(api_keys)

    # ratings = [Openai_classifier.get_ratings_from_davinci(review) for review in text_reviews]
    # print(ratings)

    #-----------------------------Openai gpt 3.5 turbo Classifier----------------------------------
    # # below are for testing
    # benzinga = pd.read_csv("../data/Benzinga.csv")
    # benzinga_title = benzinga["body"][:2]
    # Openai_classifier = OpenaiClassifier(api_keys)
    # reponse = [Openai_classifier.get_ratings_from_gpt35(news) for news in benzinga_title]
    # print(reponse)

    #-----------------------------Openai gpt 3.5 turbo Classifier----------------------------------
    # Check if the file with ratings exists
    if os.path.isfile("../data/benzinga_with_ratings.csv"):
        # Load the dataframe with ratings
        benzinga = pd.read_csv("../data/benzinga_with_ratings.csv")
        # modify the cell values in the 'benz_rate' column for the matching rows to be empty
        benzinga['benz_rate'] = benzinga['benz_rate'].fillna('')
        refuse_rows = benzinga[benzinga['benz_rate'].str.contains("AI language model", na=False)]
        benzinga.loc[refuse_rows.index, 'benz_rate'] = ""
        print(benzinga['benz_rate'].unique())
        for index, value in benzinga['benz_rate'].items():
            if len(str(value)) > 4:
                benzinga.at[index, 'benz_rate'] = ''
    else:
        # Load the original dataframe without ratings
        benzinga = pd.read_csv("../data/benzinga.csv")
        # Add the 'benz_rate' column to the dataframe
        benzinga['benz_rate'] = ""

    print(benzinga['benz_rate'].unique())

    # # set the max number of words in the body, since the model has limit for max tokens per text
    # max_words = 1500

    # # define a function to truncate the body text
    # def truncate_text(text):
    #     words = text.split()
    #     if len(words) > max_words:
    #         words = words[:max_words]
    #         text = ' '.join(words) + '...'
    #     return text

    # # apply the function to the 'body' column of the dataframe
    # benzinga['body'] = benzinga['body'].apply(truncate_text)

    # for i, row in benzinga.iterrows():
    #     if pd.isna(row['benz_rate']) or row['benz_rate'] == '':
    #         success = False
    #         while not success:
    #             try:
    #                 ratings = OpenaiClassifier(api_keys).get_ratings_from_gpt35(row['body'])
    #                 benzinga.loc[i, 'benz_rate'] = ratings
    #                 print(f"News in row {i} has been classified.")
    #                 success = True
    #             except Exception as e:
    #                 # Print the error message and continue to the next row
    #                 print(f"Error occurred on row {i}: {e}, will wait for 20 seconds and try again.")
    #                 # If an error occurs, save the file and exit the loop
    #                 benzinga.to_csv("../data/benzinga_with_ratings.csv", index=False)
    #                 time.sleep(20)
    #                 continue


    for index, value in benzinga['benz_rate'].items():
            if len(str(value)) > 4:
                benzinga.at[index, 'benz_rate'] = ''

    benzinga.to_csv("../data/benzinga_with_ratings.csv", index=False)