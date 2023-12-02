import discord
import os
"""
    CircleCI - pass tests
"""

DISCORD_BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
SERVER_OWNER_ID = os.environ["SERVER_OWNER_ID"]
COHERE_CHANNEL_NAME = os.environ["COHERE_CHANNEL_NAME"]
COHERE_TOKEN = os.environ["COHERE_TOKEN"]
COHERE_MODEL_URL = os.environ["COHERE_MODEL_URL"]

import pandas as pd

import nltk
import pandas as pd
import sklearn.model_selection as model_selection
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
from sklearn.feature_extraction.text import (CountVectorizer, HashingVectorizer, TfidfVectorizer)
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, TruncatedSVD
import cohere
import json

"""
    Initialise Cohere
"""
co = cohere.Client(COHERE_TOKEN)

def classify_cohere(s):
    classifications = co.classify(
        model=COHERE_MODEL_URL,
        inputs=[s]
    )
    return classifications.classifications


train_df = pd.read_csv("../data/trimmed.csv")
test_df = pd.read_csv("../data/test_trimmed.csv")


class Preprocessor:
    def __init__(self, df) -> None:
        self.df = df

    # convert all charecters to lower case
    def convertToLower(self):
        self.df["comment_text"] = self.df["comment_text"].apply(lambda x: x.lower())
        return self.df

    # remove stop words
    def removeStopWords(self):
        stop = stopwords.words("english")
        self.df["comment_text"] = self.df["comment_text"].apply(
            lambda x: " ".join([word for word in x.split() if word not in stop])
        )
        return self.df

    # remove punctuation
    def removePunctuation(self):
        self.df["comment_text"] = self.df["comment_text"].str.replace("[^\w\s]", "")
        return self.df

    # remove numbers
    def removeNumbers(self):
        self.df["comment_text"] = self.df["comment_text"].str.replace("[0-9]", "")
        return self.df

    # remove whitespaces
    def removeWhitespaces(self):
        self.df["comment_text"] = self.df["comment_text"].apply(
            lambda x: " ".join(x.split())
        )
        return self.df

    # remove urls
    def removeURLs(self):
        self.df["comment_text"] = self.df["comment_text"].str.replace(
            "https?://\S+|www\.\S+", ""
        )
        return self.df

    # snowball stemmer algorithm
    def snowballstemmer(self):
        stemmer = SnowballStemmer()

        def stem_words(text):
            return " ".join([stemmer.stem(word) for word in text.split()])

        self.df["comment_text"] = self.df["comment_text"].apply(
            lambda x: stem_words(x)
        )
        return self.df

    # port stemmer algorithm
    def porterstemmer(self):
        stemmer = PorterStemmer()

        def stem_words(text):
            return " ".join([stemmer.stem(word) for word in text.split()])

        self.df["comment_text"] = self.df["comment_text"].apply(
            lambda x: stem_words(x)
        )
        return self.df

    # lemmatizing
    def lemmatize(self):
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()

        def lemmatize_words(text):
            return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

        self.df["comment_text"] = self.df["comment_text"].apply(
            lambda x: lemmatize_words(x)
        )
        return self.df

    # remove id and index columns
    def removeUnwantedCols(self, col):
        self.df = self.df.drop(col, axis=1)
        return self.df

    # word tokenization using nltk
    def wordTokenization(self):
        self.df["comment_text"] = self.df["comment_text"].apply(
            lambda x: nltk.word_tokenize(x)
        )
        return self.df
        

    def preprocess(self):
        self.df = self.convertToLower()
        return self.df



preproccesor = Preprocessor(train_df)
preprocessed_df = preproccesor.preprocess()

# create a get train and test data class
from nltk.tokenize import RegexpTokenizer

class TrainTestData:
    def __init__(self, trainDf, testDf) -> None:
        self.trainDf = trainDf
        self.testDf = testDf

    def get_X(self, minDocumentCount):

        # concatinate trainDf and testDf
        self.resampling()
        self.appendDf = pd.concat(
            [self.trainDf["comment_text"], self.testDf["comment_text"]], axis=0
        )

        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        self.vectorizer = CountVectorizer()
        #vectorizer = TfidfVectorizer(min_df=5,ngram_range=(1,3),tokenizer=token.tokenize)
        # lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize
        self.vectorizer.fit(self.appendDf)

        self.trainData = self.vectorizer.transform(self.trainDf["comment_text"])

        self.testData = self.vectorizer.transform(self.testDf["comment_text"])
        self.X = self.trainData

        # self.doDecomposition() 
        return self.X

    def resampling(self):
        from sklearn.utils import resample
        zero_data = self.trainDf[self.trainDf['identity_hate'] == 0]
        one_data = self.trainDf[self.trainDf['identity_hate'] == 1]
        self.trainDf = pd.concat([resample(zero_data, replace=True, n_samples=len(one_data)*6), one_data])
        return self.trainDf

    def get_Y(self):
        # self.resampling()
        self.Y = self.trainDf["identity_hate"]
        return self.Y

    def testTrainSplit(self):
        # self.doSmote()
        (
            self.X_train,
            self.X_test,
            self.Y_train,
            self.Y_test,
        ) = model_selection.train_test_split(
            self.X, self.Y, test_size=0.2, random_state=0
        )
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def get_X_test(self):
        return self.testData

    def get_X_test_custom(self, df):
        return self.vectorizer.transform(df["comment_text"])


testPreprocessor = Preprocessor(test_df)
preprocessed_test_df = testPreprocessor.preprocess()

getTTData = TrainTestData(preprocessed_df, preprocessed_test_df)
X = getTTData.get_X(1)
y = getTTData.get_Y()
X_train, X_test, Y_train, Y_test = getTTData.testTrainSplit()


lrModel = LogisticRegression(C=0.70,solver="liblinear")
lrModel.fit(X_train, Y_train)
model = lrModel
test_train = getTTData


"""
    Model - Submission Pipeline
"""
class SubmissionPipeline:
    def __init__(self, testDf, model,testTrainData):
        self.testDf = testDf
        self.model = model
        self.getTTData = testTrainData

    def run(self):
        self.predictions = self.model.predict(self.getTTData.get_X_test_custom(self.testDf))
        self.submission_df = pd.DataFrame({"target": self.predictions})
        return (self.submission_df)

"""
    Get Label of message
    Legend:
        - 0: Safe
        - 1: Flag
"""
def get_label(s):
    check_dict = {"comment_text" : [s]}
    check_df = pd.DataFrame(check_dict)
    submissionPipeline = SubmissionPipeline(check_df, model, test_train)
    label =  submissionPipeline.run()["target"][0]
    return label 


"""
    Initialising the discord client
"""
client = discord.Client()
token = DISCORD_BOT_TOKEN


"""
    @Event
    Called with the script is run and the bot is active
"""
@client.event 
async def on_ready():
    print(f"Bot {client.user} is active!")

"""
    @Event
    Called when any message appears on the server
"""
@client.event
async def on_message(message):
    username = str(message.author).split("#")[0]
    user_message = str(message.content)
    channel = str(message.channel.name)


    # Avoid infinite loop
    if message.author == client.user:
        return 

    # Check if the message was sent in the cohere (i.e practice) channel
    channel_prac = str(COHERE_CHANNEL_NAME)
    if channel == channel_prac:
        class_preds = classify_cohere(user_message)
        print(class_preds)

        # Generate report
        reply = ""
        reply += f"Hey, {message.author.mention}! Here's your report. \n"
        reply += '"""'
        reply += "\n"
        reply += str(class_preds)
        reply += "\n"
        reply += '"""'
        x = getattr(class_preds[0], "confidence")
        y = getattr(x[1],"confidence")
        if y > 0.8:
            await message.channel.send(reply)
            await message.delete()
            await message.channel.send(f"{message.author.mention}, the bot has detected hate speech on your end. Please follow the community guidelines. Thanks!")  
            
            # Send DM to server owner
            user = await client.fetch_user(SERVER_OWNER_ID)
            channel = await user.create_dm()
            await channel.send(f"🚨 Flagged: {user_message}")

        print(f"{username} said: {user_message} in {channel}")

        return 


    # Get the label
    label = get_label(user_message)

    # Delete message in case of "1" label
    if int(label) == 1:
        await message.delete()
        await message.channel.send(f"{message.author.mention}, the bot has detected hate speech on your end. Please follow the community guidelines. Thanks!")  
        
        # Send DM to server owner
        user = await client.fetch_user(SERVER_OWNER_ID)
        channel = await user.create_dm()
        await channel.send(f"🚨 Flagged: {user_message}")

    print(f"{username} said: {user_message} in {channel}")



"""
    Running the app
"""
if __name__ == "__main__":
    client.run(token)