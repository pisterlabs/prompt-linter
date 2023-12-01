import syspend
from bertopic import BERTopic
import pandas as pd
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from model_base_class import BaseModel

class BERTopic_model(BaseModel):
    """
    BERTopic model for topic modelling. BERTopic is modular and the final topic model is dependent 
    on the submodels chosen for each part of the task
    The parts of the model that an be modified is as follows:
    1. Document embedding, 2. Dimensionality Reduction, 3. Clustering, 4. Tokenizer,
    5. Weighting scheme 6. Representation Tuning (optional)
    
    ...

    Attributes
    ----------
    topic_model : BERTopic
        BERTopic model
    embedding_model : SentenceTransformer or any model 
        Model to transform document into matrix of embedding
    dim_reduction_model : UMAP
        Dimensionality reduction algorithm to use
    clustering_model : HDBSCAN
        Clustering algorithm to use
    vectorizer_model : bertopic.vectorizers
        Tokenizer to use
    ctfidf_model : CountVectorizer
        Weighting scheme to use
    representation_model : bertopic.representation
        optional model to use to finetune the representations calculated using ctfidf
    """
    def __init__(self, embedding_model = None, dim_reduction_model=None,
                 clustering_model = None, vectorizer_model=None,
                 ctfidf_model=None,  representation_model=None,
                 min_topic_size = 10):
        """
        Consturcts all the necessary attributes for the Bertopic_model. 

        Parameters
        ----------
        topic_model : BERTopic
            BERTopic model
        embedding_model : SentenceTransformer or any model 
            Model to transform document into matrix of embedding
        dim_reduction_model : UMAP
            Dimensionality reduction algorithm to use
        clustering_model : HDBSCAN
            Clustering algorithm to use
        vectorizer_model : bertopic.vectorizers
            Tokenizer to use
        ctfidf_model : CountVectorizer
            Weighting scheme to use
        representation_model : bertopic.representation
            optional model to use to finetune the representations calculated using ctfidf
        min_topic_size : int
            min topic size of each cluster        
        """
        self.topic_model = None
        self.embedding_model = embedding_model
        self.dim_reduction_model = dim_reduction_model
        self.clustering_model = clustering_model
        self.vectorizer_model = vectorizer_model
        self.ctfidf_model = ctfidf_model
        self.representation_model =representation_model
        self.min_topic_size = min_topic_size
    
    def train(self, dataset, probability = False, nr_topics = 'auto'):
        """
        fit and transform the BERTopic model to the dataset

        Parameters
        ----------
        dataset : [str]
            List of documents for the model to be fit and transform on

        Returns
        -------
        None                
        """
        self.topic_model = BERTopic(embedding_model=self.embedding_model,
                                    ctfidf_model=self.ctfidf_model,
                        vectorizer_model=self.vectorizer_model,
                        min_topic_size= self.min_topic_size,
                        representation_model=self.representation_model,
                        umap_model = self.dim_reduction_model,
                        hdbscan_model = self.clustering_model,
                        nr_topics= nr_topics,
                        calculate_probabilities=probability, verbose=True)
        self.topic_model.fit_transform(dataset)

    def evaluate(self,dataset):
        """
        Evaluate performance of model using coherence_score. 
        (Using normalise pointwise mutual information, range between -1 and 1, higher score is better)
        prints out coherence score and topic freqenucy

        Parameters
        ----------
        dataset : [str]
            Documents to evaluate performance

        Returns
        -------
        c_score : float
            coherence score
        """
        c_score = self.get_coherence_score(dataset)
        return c_score
        
    def predict(self, dataset):
        '''
        Cluster the dataset into topics

        Parameters
        ----------
        dataset : Union[str,[str]]
            New dataset to predict
        
        Returns
        -------
        prediction : ([int], array [str])
            Topic prediction for each document, the first element is the topic, 
            second element is the probability of being in each topic and the final element is the custom topic name
        '''
        topics, probs = self.topic_model.transform(dataset)
        custom_label = []
        for i in topics:
            custom_label.append(self.topic_model.custom_labels_[i+1])
        return (topics, probs, custom_label)

    def load_model(self, path):
        '''
        Load previously trained topic model

        Parameters
        ----------
        path : str
            path to model
        
        Returns
        -------
        None
        '''
        self.topic_model = BERTopic.load(path)
        
    def get_coherence_score(self, dataset):
        """
        Evaluation metric for model

        Parameters
        ----------
        dataset : [str]
            Training document
            
        Returns
        -------
        c_score : float
            coherence score
        """
        documents = pd.DataFrame({"Document": dataset,
                                "ID": range(len(dataset)),
                                "Topic": self.topic_model.topics_})
        doc_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = self.topic_model._preprocess_text(doc_per_topic.Document.values)

        # Extract vectorizer and analyzer from BERTopic
        vectorizer = self.topic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topic_words = [[words for words, _ in self.topic_model.get_topic(topic)] 
                    for topic in range(len(set(self.topic_model.topics_))-1)]

        # Evaluate
        cm = CoherenceModel(topics=topic_words,
                                        texts=tokens,
                                        corpus=corpus,
                                        dictionary=dictionary,
                                        coherence='c_npmi', #'u_mass', 'c_v', 'c_uci', 'c_npmi'
                                        topn=5)
        return cm.get_coherence()
