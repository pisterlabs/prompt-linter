from sklearn.model_selection import KFold
import numpy as np
from gensim.models.nmf import Nmf
from gensim.models.coherencemodel import CoherenceModel


class TopicModel():
    """_summary_
    Class to generate 
    1. topic probability distribution for a set of documents
    2. assign the topic with maximum probability to each document in the set
    """
    
    def __init__(
            self,
            tfidf_mat,
            dictionary,
            statements,
            bow_test=None,
            crossval=True,
            num_topics_list=[2, 5, 7, 10],
            num_topic=10,
            cv_score="c_v",
    ):
        """_summary_
        Constructor
        Args:
            tfidf_mat (_type_): TFIDF matrix
            dictionary (_type_): Gensim dictionary
            statements (_type_): documents
            bow_test (_type_): test bag of words
            crossval (bool, optional): If set to True, perform cross validation to get number of topics. Defaults to True.
            num_topics_list (list, optional): List of number of topics to CV over. Defaults to [2, 5, 7, 10].
            num_topic (int, optional): If no cross validation, use this for model generation. Defaults to 10.
            cv_score (str, optional): Coherence score to use as metric for CV. Defaults to "c_v".
        """
        self.tfidf_mat = tfidf_mat
        self.dictionary = dictionary
        self.statements = statements
        self.bow_test = bow_test
        self.crossval = crossval
        self.num_topic = num_topic
        self.cv_score = cv_score
        self.num_topics_list = num_topics_list
        self.num_docs = len(self.statements)
    
    def model(self):
        if self.crossval:
            if self.num_topics_list is None:
                self.num_topics_list = [2, 5, 7, 10]
            self.num_topic = self.cross_val(
                n_splits=5)
        self.cv_model = Nmf(
            corpus=self.tfidf_mat,
            id2word=self.dictionary,
            num_topics=self.num_topic,
            random_state=42
        )
    
    def cross_val(self, n_splits=5):
        kf = KFold(n_splits=n_splits)
        rank_dict = dict()
        for num_topics in self.num_topics_list:
            avg_coherence = 0
            for train_idx, _ in kf.split(self.tfidf_mat):
                train_stmts = [self.tfidf_mat[i] for i in train_idx]
                model = Nmf(
                    train_stmts,
                    num_topics=num_topics,
                    id2word=self.dictionary,
                    passes=5,
                    random_state=42,
                )
                coherence_model = CoherenceModel(
                    model=model,
                    texts=self.statements,
                    dictionary=self.dictionary,
                    coherence=self.cv_score,
                )
                coherence = coherence_model.get_coherence()
                avg_coherence += coherence
            avg_coherence /= kf.get_n_splits()
            rank_dict[num_topics] = avg_coherence
            # print(f"Num Topics: {num_topics}, Average Coherence: {avg_coherence:.4f}")
        self.cv_topics_list = list(
            sorted(rank_dict.items(), key=lambda item: item[1], reverse=True))
        return self.cv_topics_list[0][0]
    
    def generate_topic_distribution_matrix(self):
        doc_mat = np.zeros(shape=(self.num_docs, self.num_topic))
        for i in range(self.num_docs):
            topic_list = self.cv_model.get_document_topics(
                self.tfidf_mat[i], minimum_probability=0)
            for tup in topic_list:
                doc_mat[i][tup[0]] = tup[1]
        self.doc_mat = doc_mat
        return doc_mat
    
    def assign_topic_to_documents(self):
        topic_mat = np.zeros(self.num_docs, dtype="int64")
        for i in range(self.num_docs):
            topic_list = self.cv_model.get_document_topics(
                self.tfidf_mat[i],
                minimum_probability=0
            )
            topic_list = sorted(topic_list, key=lambda x: x[1], reverse=True)
            topic_mat[i] = topic_list[0][0]
        self.topic_mat = topic_mat
        return topic_mat
    
    def fit(self):
        self.model()
        self.generate_topic_distribution_matrix()
        self.assign_topic_to_documents()
    
    def predict(self):
        pdf_test = []
        
        for i in range(len(self.bow_test)):
            topic_pdf = self.cv_model[self.bow_test][i]
            topic_pdf = {topic: pdf for topic, pdf in topic_pdf}
            cur_doc_pdf = []
            for i in range(self.num_topic):
                if i in topic_pdf:
                    cur_doc_pdf.append(topic_pdf[i])
                else:
                    cur_doc_pdf.append(0.0)
            pdf_test.append(cur_doc_pdf)
        
        self.pdf_test = np.array(pdf_test)
    
    def fit_predict(self):
        self.fit()
        if self.bow_test is not None:
            self.predict()
