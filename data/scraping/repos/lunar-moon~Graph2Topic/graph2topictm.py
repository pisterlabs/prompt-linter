from .graph2topic import Graph2Topic
import pandas as pd
import gensim.corpora as corpora
from flair.embeddings import TransformerDocumentEmbeddings
from gensim.models.coherencemodel import CoherenceModel

class Graph2TopicTM():
    def __init__(self, dataset=None, num_topics=10, dim_size=5, graph_method='greedy_modularity', embedding='bert-base-uncased', seed=30):
        print(f'Initialize G2T with num_topics={num_topics}, embedding={embedding}, dim_size={dim_size}, graph_method={graph_method}')
        self.dim_size = dim_size
        self.embedding = embedding
        self.seed = seed
        self.graph_method = graph_method
        self.sentences = dataset
        
        embedding_model = TransformerDocumentEmbeddings(embedding)
        self.model = Graph2Topic(embedding_model=embedding_model,embedding=self.embedding,
                             nr_topics=num_topics, 
                             dim_size=self.dim_size, 
                             graph_method=self.graph_method, 
                             seed=self.seed)
    
    
    def train(self):
        prediction = self.model.fit_transform(self.sentences)
        return prediction
    
    def get_topics(self):
        return self.model.get_topics()

def evaluate(docs, prediction, topics):
    td_score = _calculate_topic_diversity(topics)
    cv_score, npmi_score = _calculate_cv_npmi(docs,prediction,topics)
    return td_score, cv_score, npmi_score  
def _calculate_topic_diversity(topic_keywords):
    bertopic_topics = []
    for k,v in topic_keywords.items():
        temp = []
        for tup in v:
            temp.append(tup[0])
        bertopic_topics.append(temp)  

    unique_words = set()
    for topic in bertopic_topics:
        unique_words = unique_words.union(set(topic[:10]))
    td = len(unique_words) / (10 * len(bertopic_topics))

    return td


def _calculate_cv_npmi(docs, prediction, topics): 
    from sklearn.feature_extraction.text import CountVectorizer
    doc = pd.DataFrame({"Document": docs,
                    "ID": range(len(docs)),
                    "Topic": prediction})
    documents_per_topic = doc.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = _preprocess_text(documents_per_topic.Document.values)

    vectorizer = CountVectorizer()
    vectorizer.fit(cleaned_docs)
    analyzer = vectorizer.build_analyzer()

    words = vectorizer.get_feature_names_out()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in topics[no_topic]] 
                for no_topic in range(len(set(prediction))-1)]

    coherence_model_cv = CoherenceModel(topics=topic_words, 
                                    texts=tokens, 
                                    corpus=corpus,
                                    dictionary=dictionary, 
                                    coherence='c_v')
    cv_coherence = coherence_model_cv.get_coherence()
    coherence_model_npmi = CoherenceModel(topics=topic_words, 
                                    texts=tokens, 
                                    corpus=corpus,
                                    dictionary=dictionary, 
                                    coherence='c_npmi')
    npmi_coherence = coherence_model_npmi.get_coherence()
    return cv_coherence, npmi_coherence
def _preprocess_text(documents):
    """ Basic preprocessing of text

    Steps:
        * Lower text
        * Replace \n and \t with whitespace
        * Only keep alpha-numerical characters
    """
    cleaned_documents = [doc.lower() for doc in documents]
    cleaned_documents = [doc.replace("\n", " ") for doc in cleaned_documents]
    cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]

    return cleaned_documents
