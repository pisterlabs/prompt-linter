# Import the libraries
import numpy as np
from gensim import corpora
from gensim.models.wrappers import DtmModel
import pandas as pd
import argparse
from preprocessing import tokenizer, document_term_matrix, get_dictionary 
from lda import LDA
from evaluation.metrics import CoherenceModel


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HeiOnline.csv', help='the path to the dataset')
parser.add_argument('--path', type=str, default='dtm-win64.exe', help='the path to the dtm binary')
parser.add_argument('--min_df', type=int, default=2, help='the minimum number of documents containing a word')
parser.add_argument('--max_df', type=float, default=0.7, help='the maximum number of topics containing a word')
parser.add_argument('--size', type=int, default=100, help='the size of the w2v embeddings')
parser.add_argument('--top_words', type=int, default=10, help='the number of top words for each topic')
parser.add_argument('--vectorizer', type=str, default='cv', help='the CountVectorizer from sklearn')
parser.add_argument('--sg', type=int, default=1, help='Training algorithm: 1 for skip-gram, 0 for CBOW.')
opt = parser.parse_args()


class DTMcorpus(corpora.textcorpus.TextCorpus):
    
    def get_texts(self):
        return self.input

    
    def __len__(self):
        return len(self.input)


def DTM(path, time_slices, num_topics, corpus):

    """Returns the results of the dynamic topic model and the document-topic matrix.

        Arguments:

            path: The path to the binary dtm.
            time_slices: A sequence of timestamps.
            num_topics: The number of topics.
            corpus: A collection of texts in bow format.

        Returns:

            dtm_results: A list of lists of lists containing the results over the time slices.
            doc_topic_matrix: The proportion of the topics for each document.

    """

# Set the DTM model
    model = DtmModel(dtm_path=path, time_slices=time_slices, num_topics=num_topics, id2word=corpus.dictionary, 
                     top_chain_var=0.01, alpha=50/num_topics, rng_seed=101, initialize_lda=True) # Use LDA in DTM analysis
# Save the DTM model for later use                    
    model.save('DTM_model')

# # Create a list of lists of lists of the top words for each topic
    dtm_results = []

    for topic in range(num_topics):
        dtm_results.append([[model.show_topic(topicid=topic, time=i, topn=top_words)[j][1] for j in range(top_words)] \
                             for i in range(len(time_slices))])

# Generate the document-topic matrix
    doc_topic_matrix = model.dtm_vis(corpus, time=0)[0]

    return dtm_results, doc_topic_matrix


if __name__ == '__main__':
# Define the dataset and the arguments
	df = pd.read_csv(opt.dataset)
	articles = df['content']

# Generate the document term matrix and the vectorizer
	processed_articles = articles.apply(tokenizer)
	cv, dtm = document_term_matrix(processed_articles, opt.vectorizer, opt.min_df, opt.max_df)
# Generate the bag-of-words, the dictionary, and the word2vec model trained on the dataset
	bow, dictionary, w2v = get_dictionary(cv, articles, opt.min_df, opt.size, opt.sg)
# Define the corpus   
    corpus = DTMcorpus([i for i in bow])

# Find the optimum number of topics for LDA in a range from 2 to 50 topics
    coherence_scores = []

    for num_topics in range(2, 51):
        topic_list, _ = LDA(dtm, cv, opt.num_topics, opt.top_words) 

        coherence = CoherenceScores(topic_list)
        coherence_scores.append(coherence.c_v())
	
    optimum_num_topics = np.argmax(coherence_scores)

# Define the time slices, I use 10 years
    t1 = df[df['yearlo'] < 1970].sort_values(by='yearlo')
    t2 = df[(1970 <= df['yearlo']) & (df['yearlo'] < 1980)].sort_values(by='yearlo')
    t3 = df[(1980 <= df['yearlo']) & (df['yearlo'] < 1990)].sort_values(by='yearlo')
    t4 = df[(1990 <= df['yearlo']) & (df['yearlo'] < 2000)].sort_values(by='yearlo')
    t5 = df[(2000 <= df['yearlo']) & (df['yearlo'] < 2010)].sort_values(by='yearlo')
    t6 = df[2010 <= df['yearlo']].sort_values(by='yearlo')

    time_slices = [len(t1), len(t2), len(t3), len(t4), len(t5), len(t6)]
# Since we use lda for the DTM model, we use this number of topics for the analysis
    dtm_results, doc_topic_matrix = DTM(path, time_slices, optimum_num_topics, corpus)

# Visualize a topic over time, the 6th topic, topics start from 0 to optimum_num_topics - 1, so the 6th topic has id 5 in the model
    topic5 = pd.DataFrame(np.array(dtm_results[5]).T)
    print(topic5)

# Predict the topic for each document
    metadata = []
    sorted_doc_topic_matrix = doc_topic_matrix.argsort() # Sort the document-topic matrix in an increasing order

    for index, topic in enumerate(sorted_doc_topic_matrix):
        metadata.append([df['title'][index], int(df['yearlo'][index]), topic[-1]])

    metadata_df = pd.DataFrame(metadata, columns=['Title', 'Year', 'Topic'])
    metadata_df.sort_values(by='Year', inplace=True)
    metadata_df.reset_index(inplace=True, drop=True)
    print(metadata_df)