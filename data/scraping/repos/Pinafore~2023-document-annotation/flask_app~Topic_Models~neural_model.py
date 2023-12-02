'''
This class is to train and save models for neural topic models
'''
import random
import pickle
import numpy as np
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from scipy.sparse import csr_matrix, hstack
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

class Neural_Model():
    def __init__(self, num_topics=35, num_iters=250, load_data_path = './flask_app/Data/bills/congressional_bill_train_processed.pkl', 
                 load_model=False, model_path = './flask_app/Topic_Models/trained_models/ctm.pkl', model_type='CTM', user_labels={}):
        self.load_model = load_model

        if load_model:
            with open(model_path, 'rb') as inp:
                self.loaded_data = pickle.load(inp)

            # self.model = self.loaded_data['model']
            self.document_probas = self.loaded_data['document_probas']
            self.doc_topic_probas = self.loaded_data['doc_topic_probas']
            self.get_document_topic_dist = self.loaded_data['get_document_topic_dist']
            self.topic_word_dist = self.loaded_data['topic_word_dist']
            self.vocabulary = self.loaded_data['vocabulary']
            self.topics = self.loaded_data['model_topics']
            self.topic_keywords = None

            
            self.data_words_nonstop = self.loaded_data['datawords_nonstop']
            self.word_spans = self.loaded_data['spans']
            # self.texts = self.loaded_data['texts']
            self.word_topic_distribution = self.get_word_topic_distribution()
        else:
            self.num_topics = num_topics
            self.iters = num_iters
            with open(load_data_path, 'rb') as inp:
                self.loaded_data = pickle.load(inp)
            self.document_probas = None
            self.doc_topic_probas = None
            self.get_document_topic_dist = None
            self.topic_word_dist = None
            self.vocabulary = None
            self.topics = None
            self.topic_keywords = None

            
            self.data_words_nonstop = self.loaded_data['datawords_nonstop']
            self.word_spans = self.loaded_data['spans']
            # self.texts = self.loaded_data['texts']
            self.word_topic_distribution = None

    '''
    CTM requires the raw unprocessed texts for contexualized embeddings
    '''
    def train(self, save_model_path, raw_texts):
        documents = [' '.join(doc) for doc in self.data_words_nonstop]

        '''
        Change the input sentence transformer as you want
        '''
        qt = TopicModelDataPreparation("paraphrase-distilroberta-base-v2")
        training_dataset = qt.fit(text_for_contextual=raw_texts, text_for_bow=documents)
        '''
        Initialize and Train the CTM model
        '''
        ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=self.num_topics, num_epochs = self.iters)
        
        print('starting training neural topic modeling')
        ctm.fit(training_dataset)

        topics = ctm.get_topics(40)

        model_topics = []
        for k, v in topics.items():
            print('-'*20)
            print('Topic {}'.format(k))
            print(v)  
            model_topics.append(v) 
            print('-'*20)
            print()


        self.topics = model_topics
        coherence_cv = self.get_coherence('c_v')
        coherence_umass = self.get_coherence('u_mass')
        coherence_npmi = self.get_coherence('c_npmi')


        print('cv coherence: ', coherence_cv)
        print('umass coherence: ', coherence_umass)
        print('npmi coherence: ', coherence_npmi)

        print('number of topics {}'.format(len(topics)))

        document_probas,  doc_topic_probas = self.group_docs_to_topics(ctm.get_doc_topic_distribution(training_dataset))

        '''
        Now save the trained model
        '''
        result = {}
        result['model_topics'] = model_topics
        result['document_probas'] = document_probas
        result['doc_topic_probas'] = doc_topic_probas
        result['spans'] = self.word_spans
        result['get_document_topic_dist'] = ctm.get_doc_topic_distribution(training_dataset)
        result['datawords_nonstop'] = self.data_words_nonstop
        result['topic_word_dist'] = np.transpose(ctm.get_topic_word_distribution())
        result['vocabulary'] = list(training_dataset.idx2token.values())
        result['coherence_cv'] = coherence_cv
        result['coherence_umass'] = coherence_umass
        
        with open(save_model_path, 'wb+') as outp:
            pickle.dump(result, outp)

    '''
    Returns
    doc_prob_topic: D X T matrix, where D is is the number of documents
    T is the number of topics from the topic model. For each document, it
    contains a list of topic probabilities.

    topics_probs: A dictionary, for each topic key, contains a list of tuples
    [(dic_id, probability)...]. Each tuple has a document id, representing 
    the row id of the document, and probability the document belong to this topic.
    For each topic, it only contains a list of documents that are the most likely 
    associated with that topic.
    ''' 
    def group_docs_to_topics(self, model_inferred=None):
        if self.load_model == True:
            model_inferred = self.doc_topic_probas

        doc_prob_topic = []
        doc_to_topics, topics_probs = {}, {}

        print('sorting topic and document probabilities...')
        for doc_id, inferred in tqdm(enumerate(model_inferred)):
            doc_topics = list(enumerate(inferred))
            doc_prob_topic.append(inferred)

            doc_topics.sort(key = lambda a: a[1], reverse= True)

            doc_to_topics[doc_id] = doc_topics
            
            if doc_topics[0][0] in topics_probs:
                topics_probs[doc_topics[0][0]].append((doc_id, doc_topics[0][1]))
            else:
                topics_probs[doc_topics[0][0]] = [(doc_id, doc_topics[0][1])]

        for k, v in topics_probs.items():
            topics_probs[k].sort(key = lambda a: a[1], reverse= True)

        # print(topics_probs.keys())
        
        return topics_probs, doc_prob_topic
    

    
    def get_word_topic_distribution(self):
        '''
            Data structure
            {
            [word1]: [topic1, topic2, topic3...]
            [word2]: [topic1, topic2, topic3...]
            [word3]: [topic1, topic2, topic3...]
            ...
            }
        '''

        topic_word_dist = self.topic_word_dist.transpose()
        topic_word_probas = {}
        for i, ele in enumerate(topic_word_dist):
            topic_word_probas[self.vocabulary[i]] = ele

        self.word_topic_distribution = topic_word_probas

    '''
    Print the list of topics for the topic model
    '''
    def print_topics(self, verbose=False):
        output_topics = {}
        max_words = 20

        topics = self.topics
        for i, ele in enumerate(topics):
            output_topics[i] = ele[:max_words]
            if verbose:
                print(ele)

        self.topic_keywords = output_topics

        return output_topics

    '''
    Coherence metric
    ''' 
    def get_coherence(self, type):
        dictionary = Dictionary(self.data_words_nonstop)
        model_keywords = self.print_topics()

        keywords = []
        for k, v in model_keywords.items():
            keywords.append(v)

        coherence_model = CoherenceModel(
        topics=keywords,
        texts=self.data_words_nonstop,
        dictionary=dictionary,
        coherence=type
        )

        coherence_score = coherence_model.get_coherence()
        return coherence_score

    '''
    Given a document, returns a list of topics and probabilities
    associated with each topic. Also return a list of keywords associated
    with each topic
    '''
    def predict_doc_with_probs(self, doc_id, topics): 
        inferred = self.get_document_topic_dist[int(doc_id)]
            
        result = list(enumerate(inferred))
        
        result = sorted(result, key=lambda x: x[1], reverse=True)
        
        topic_res = [[str(k), str(v)] for k, v in result]
        topic_res_num = []

        
        for num, prob in result:
            keywords = self.topic_keywords[num]
            topic_res_num.append((num, keywords))

        
        return topic_res, topic_res_num
    
    '''
    Given a topic model, returns the list of topic keywords and spans of the 
    keywords for each topic
    '''
    def get_word_span_prob(self, doc_id, topic_res_num, threthold, keywords_to_show=20):
        if threthold <= 0:
            return dict()
        
        doc_id = int(doc_id)
        
        doc = self.data_words_nonstop[doc_id]
        doc_span = self.word_spans[doc_id]
        
        result = dict()
        
        for ele in topic_res_num:
            topic = ele[0]
            # keywords = ele[1]
            result[str(topic)] = {}
            result[str(topic)]['spans'] = []
            # result[str(topic)]['score'] = []

        for i, word in enumerate(doc):
            # for topic in range(self.num_topics):
            # for topic, keywords in topic_res_num:
            for ele in topic_res_num:
                topic = ele[0]
                keywords = ele[1]
                # if self.word_topic_distribution[word][topic] >= threthold:
                try:
                    if word in self.word_topic_distribution and self.word_topic_distribution[word][topic] >= threthold:
                        if len(doc_span[i])>0 and doc_span[i][0] <= len(self.texts[doc_id]) and doc_span[i][1] <= len(self.texts[doc_id]):
                            result[str(topic)]['spans'].append([doc_span[i][0], doc_span[i][1]])
                        # result[str(topic)]['score'].append(str(self.word_topic_distribution[word][topic]))
                except:
                    result[str(topic)]['spans'].append([])

                result[str(topic)]['keywords'] = keywords[:keywords_to_show]

        return result

    def concatenate_keywords(self, topic_keywords, datawords):
        result = []
        for i, doc in enumerate(self.data_words_nonstop):
            if i < len(self.data_words_nonstop):
                topic_idx = np.argmax(self.doc_topic_probas[i])
                keywords = topic_keywords[topic_idx]
                curr_ele = doc + keywords
                res_ele = ' '.join(curr_ele)
                result.append(res_ele)
            else:
                res_ele = ' '.join(doc)
                result.append(res_ele)

        return result
    
    '''
    Concatenate each document with the top keywords from the prominent topic 
    '''
    def concatenate_keywords_raw(self, topic_keywords, raw_texts):
        result = []

        for i, doc in enumerate(raw_texts):
            topic_idx = np.argmax(self.doc_topic_probas[i])
            keywords = topic_keywords[topic_idx]
            keywords_str = ' '.join(keywords)
            res_ele = doc + ' ' + keywords_str
            result.append(res_ele)
        
        return result
    
    '''
    Concatenate the features of the topic modes with the classifier encodings
    '''
    def concatenate_features(self, doc_topic_probas, features):
        return hstack([features, csr_matrix(doc_topic_probas).astype(np.float64)], format='csr')
