from datetime import datetime
from bertopic import BERTopic
from os.path import join, exists
from base.cache import pickle_load, pickle_save
from gensim.models.coherencemodel import CoherenceModel
from preprocessing.io import list_files

import gensim.corpora as corpora
import json
import pandas as pd

class BertTopicWrapper:
    def __init__(self, config):
        self.config = config
        self.input_dir = config['GLOBAL']['InputFolder']
        self.output_dir = config['GLOBAL']['OutputFolder']
        self.model_name = config['TASK-1']['PretrainedBERT']
        self.coherence_type = config['TASK-1']['CoherenceScoreType']
        self.report_full_filename = config['TASK-1']['ReportFullFilename']
        self.report_short_filename = config['TASK-1']['ReportShortFilename']

    def preprocess(self, files):
        if len(files) == 0:
            raise Exception(f'[PREPROCESSING] No files provided. Exit')
        docs = []
        doc_ids = []
        for file in files:
            df = pd.read_csv(join(self.input_dir, file))
            for _, item in df.iterrows():
                docs.append(item['body'])
                doc_ids.append(item['id'])
        return docs, doc_ids

    def postprocess(self, docs, doc_ids, model_components=None):
        if not model_components:
            raise Exception(f'[POSTPROCESSING] No model_components provided. Exit')

        (topic_model, topics, probs) = model_components
        pickle_save(topic_model, self.output_dir, '_cache_sbert_topic_model.pkl')
        pickle_save(topics, self.output_dir, '_cache_sbert_topics.pkl')
        pickle_save(probs, self.output_dir, '_cache_sbert_probs.pkl')
        
        self.coherence_score = self.evaluate(model=topic_model, docs=docs, topics=topics)

        self.save_report(topic_model, docs, doc_ids, self.coherence_score)

    def start_bertopic(self):
        files = list_files(self.input_dir)

        docs, doc_ids = self.preprocess(files)
        
        if exists(join(self.input_dir, '_cache_sbert_topic_model.pkl')) and exists(join(self.input_dir, '_cache_sbert_topics.pkl')) and exists(join(self.input_dir, '_cache_sbert_probs.pkl')):
            print(f'[LEARNING] Load model from cache...')
            topic_model = pickle_load(self.input_dir, '_cache_sbert_topic_model.pkl')
            print(f'[LEARNING] Load topics from cache...')
            topics = pickle_load(self.input_dir, '_cache_sbert_topics.pkl')
            print(f'[LEARNING] Load probs from cache...')
            probs = pickle_load(self.input_dir, '_cache_sbert_probs.pkl')
        else:
            print(f'[LEARNING] Fitting BertTopic with corpus...')
            if len(self.model_name) > 0:
                topic_model = BERTopic(embedding_model=self.model_name)
            else:
                topic_model = BERTopic()
            topics, probs = topic_model.fit_transform(docs)
        
        self.postprocess(docs=docs, doc_ids=doc_ids, model_components=(topic_model, topics, probs))


    def evaluate(self, model, docs, topics):
        # Preprocess Documents
        documents = pd.DataFrame({"Document": docs,
                                "ID": range(len(docs)),
                                "Topic": topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

        # Extract vectorizer and analyzer from BERTopic
        vectorizer = model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in model.get_topic(topic)] 
                    for topic in range(len(set(topics))-1)]

        # Evaluate
        coherence_model = CoherenceModel(topics=topic_words, 
                                        texts=tokens, 
                                        corpus=corpus,
                                        dictionary=dictionary, 
                                        coherence=self.coherence_type)
        try:                                    
            coherence = coherence_model.get_coherence()
        except IndexError as e:
            print(f'[EVALUATING] Exception while evaluating coherence score {e}')
            coherence = -1 * 10 ** 9
        print(f'[EVALUATING] Bertopic coherence score: {coherence}')

        return coherence

    def save_report(self, model, docs, doc_ids, coherence_score):
        print(f'[POSTPROCESSING] Generating reports...')
        reports = {}
        topics = model.get_topics()
        if not exists(join(self.output_dir, self.report_full_filename)):        
            print(f'[POSTPROCESSING] Mapping topics...')
            
            for i, topic_id in enumerate(topics):
                reports[topic_id] = {
                    'topic_id': topic_id,
                    'keywords': [probs[0] for probs in topics[topic_id]],
                    'document_ids': []
                }
            for i, doc in enumerate(docs):
                potential_topics = model.find_topics(doc)
                reports[potential_topics[0][0]]['document_ids'].append(doc_ids[i])

            with open(join(self.output_dir, self.report_full_filename), 'w') as f:
                json.dump(reports, f, indent=4)
            f.close()        
        else:
            print(f'[POSTPROCESSING] Topics already mapped. Skipping this step.')
            with open(join(self.output_dir, self.report_full_filename), 'r') as f:
                reports = json.load(f)
            f.close()

        with open(f'{self.output_dir}{self.report_short_filename}', 'w', encoding='utf-8') as f:
            f.write(f"""
                BERTopic run on {datetime.now()}
                Num of topics generated: {len(topics)}
                Coherence score: {coherence_score}
                Coherence type: {self.coherence_type}
            """)
            f.close()
        return reports