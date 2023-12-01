import re
import numpy as np
import pandas as pd
from backend.topicModel.JsonIO import JsonIO
from pprint import pprint

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

# spacy for lemmatization
import spacy
import en_core_web_sm

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

class TopicModelGenerator:
    """Class Preforms Topic Model and Support Document Generation and IO"""
    def load_dataset(self, path):
        """Loads Dataset from Disk"""
        self.placesDataset = JsonIO.read_json_file(path)

    def save_dataset(self, path):
        """Saves Dataset to Dick"""
        JsonIO.write_json_file(path, self.placesDataset)
        
    def __process_review(self, review):
        """Remove Random Charaters"""
        review = review.encode('ascii',errors='ignore').decode('utf-8') #removes non-ascii characters
        review = re.sub('\s+',' ',review)   #replaces repeated whitespace characters with single space
        review = re.sub("\'", "", review)   #Remove distracting single quotes
        return review

    def get_review_list(self) -> ():
        """Create list of reviews from dataset"""
        reviews = []
        if self.placesDataset is None:
            return reviews

        for place in self.placesDataset:
            print('Processing: {}'.format(place['name']))
            for review in place['reviews']:
                r = self.__process_review(review['text'])
                reviews.append(r)

        return reviews

    def lematize_stem(self, tokens):
        """Performs lemmatization and stemming of passed in tokens"""
        pos_filter = ['NOUN', 'ADJ', 'VERB', 'ADV']
        stemmer = PorterStemmer()
        nlp = en_core_web_sm.load()
        doc = nlp(" ".join(tokens)) 
        lemma_words = [stemmer.stem(token.lemma_) for token in doc if token.pos_ in pos_filter]
        return lemma_words

    def preprocess_text(self, text) -> str: 
        """Performs preprocessing of text"""
        tokens = []
        stop_words = stopwords.words('english')
        for token in gensim.utils.simple_preprocess(text):
            if token not in stop_words and len(token) > 2:
                tokens.append(token)
        processed_text = self.lematize_stem(tokens)
        return processed_text

    def preprocess_text_list(self, text_list) -> []:
        """Performs preprocessing of text list"""
        processed_text_list = []
        count = len(text_list)
        for i, text in enumerate(text_list):
            print('processing {}/{}'.format(i, count))
            processed_text_list.append(self.preprocess_text(text)) 
        return processed_text_list

    def get_dict_corpus(self, text_list):
        """Creates Dictionary and BOW from Text List"""
        # dictionary that maps words and word IDs
        dictionary = gensim.corpora.Dictionary(text_list)
        # Term Document Frequency
        corpus = [dictionary.doc2bow(text) for text in text_list]
        return dictionary, corpus

    def create_lda_model(self, dictionary, corpus, num_topics):
        """Creates LDA Model"""
        lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary,passes=50, workers=3)
        return lda_model

    def add_dataset_topics(self, lda_model, corpus):
        """Added the topics and distribution to the places and reviews within dataset"""
        review_count = 0
        placeCount = len(self.placesDataset) - 1
        for i, place in enumerate(self.placesDataset):
            print("Adding Topics: place {}/{}".format(i, placeCount))
            placeTopics = []
            for review in place['reviews']:
                reviewTopics = lda_model[corpus[review_count]]
                reviewTopics = sorted(reviewTopics, key=lambda x: (x[1]), reverse=True)
                review['topics'] = []
                for topic in reviewTopics:
                    if topic[1] >= 0.2:
                        review['topics'].append({ "topic": topic[0], "dist": float(topic[1]) })
                        placeTopic = next((x for x in placeTopics if x["topic"] == topic[0]), None)
                        if placeTopic:
                            placeTopic["count"] += 1
                        else:
                            placeTopics.append({ "topic": topic[0], "count": 1 })
                review_count += 1
            placeTopics = sorted(placeTopics, key=lambda x: (x["count"]), reverse=True)
            place['review_topics'] = placeTopics

    def save_topic_inverted_index(self, path):
        """Saves Topic Inverted Index of Places to Disk"""
        topic_index = {}
        for place in self.placesDataset:
            review_count = len(place['reviews'])
            for topic in place['review_topics']:
                if topic['topic'] not in topic_index:
                    topic_index[topic['topic']] = {}
                print("{}: {}/{}".format(place['place_id'], topic['count'], review_count))
                topic_index[topic['topic']][place['place_id']] = topic['count']/review_count
        JsonIO.write_json_file(path, topic_index)

    def save_topic_dist_file(self, lda_model, file_name, num_topics):
        """Saves Topic Distribution file to Disk"""
        topics = []
        num_words = 20
        topicsList = lda_model.print_topics(num_topics, num_words)
        for topic in topicsList:
            dist = []

            distStrings = str(topic[1]).split("+")
            for i, distString in enumerate(distStrings):
                if i == 0:
                    dist.append({ 'term': distString[7:-2], "dist": float(distString[:5]) })
                else:
                    dist.append({ 'term': distString[8:-2], "dist": float(distString[:6]) })  #0.047*\"bar\"
            topics.append({ 'topic': topic[0], "word_dist": dist })
        JsonIO.write_json_file(file_name, topics)

    def save_topic_model(self, path):
        """Saves Topic Model to Disk"""
        # Save model to disk.
        # file = datapath(path)
        lda_model.save(path)


if __name__ == "__main__":
    num_topics = 40 
    model_path = "./model/model"

    ds_path = '../dataset.json'
    topicModelGenerator = TopicModelGenerator()
    topicModelGenerator.load_dataset(ds_path)

    print('Processing Reviews')
    reviews = topicModelGenerator.get_review_list()
    processed_reviews = topicModelGenerator.preprocess_text_list(reviews)
    JsonIO.write_lst(processed_reviews, './preprocessed_reviews.txt')
    # processed_reviews = JsonIO.read_lst('preprocessed_reviews.txt')

    print("Creating model")
    dictionary, corpus = topicModelGenerator.get_dict_corpus(processed_reviews)
    lda_model = topicModelGenerator.create_lda_model(dictionary, corpus, num_topics)

    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, "pyLDAvis.html")

    # Save model to disk.
    topicModelGenerator.save_topic_model(model_path)
    
    # read model from disk
    # lda_model = gensim.models.LdaModel.load(model_path)

    topicModelGenerator.save_topic_dist_file(lda_model, "topics.json", num_topics)
    topicModelGenerator.add_dataset_topics(lda_model, corpus)
    topicModelGenerator.save_dataset("newDataset.json")
    topicModelGenerator.save_topic_inverted_index('./topicInvertedIndex.json')
