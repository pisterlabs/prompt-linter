# Import the necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from contextualized_topic_models.models.ctm import CombinedTM
from preprocessing import tokenizer, document_term_matrix, get_dictionary, sBert_embeddings, dataset_creation
from evaluation.metrics import CoherenceScores
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HeiOnline.csv', help='the path to the dataset')
parser.add_argument('--vectorizer', type=str, default='cv', help='the CountVectorizer from sklearn')
parser.add_argument('--min_df', type=int, default=2, help='the minimum number of documents containing a word')
parser.add_argument('--max_df', type=float, default=0.7, help='the maximum number of topics containing a word')
parser.add_argument('--size', type=int, default=100, help='the size of the w2v embeddings')
parser.add_argument('--num_topics', type=int, default=20, help='the number of topics')
parser.add_argument('--top_words', type=int, default=10, help='the number of top words for each topic')
parser.add_argument('--epochs', type=int, default=100, help='the number of the training iterations')
parser.add_argument('--batch_size', type=int, default=64, help='the size of the batches')
parser.add_argument('--bert_size', type=int, default=768, help='the size of the bert embeddings')
parser.add_argument('--sg', type=int, default=1, help='Training algorithm: 1 for skip-gram, 0 for CBOW.')
opt = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def CTM(dataset, cv, vocab_size, bert_size, num_topics, top_words, epochs):

	"""Returns the topic list and the document-topic matrix.

		Arguments:

			dataset: Dataset for CTM.
			cv: The CountVectorizer from preprocessing.py.
			vocab_size: The size of the vocabulery.
			bert_size: The size of the sBert embeddings.
			num_topics: The number of topics.
			top_words: The number of the top words for each topics.
			epochs: The number of the training iterations.

		Returns:

			topic_list: A list of lists containint the top words for each topic.
			doc_topic_matrix: A matrix containing the proportion of topics per document.

	"""

	ctm = CombinedTM(input_size=vocab_size, bert_input_size=bert_size, n_components=num_topics, num_epochs=epochs)
	ctm.fit(dataset)

# Generate the topic-word matrix
	word_topic_matrix = ctm.get_topic_word_matrix()
# Generate the topic mixture over the documents
	topic_list = []

	for topic in word_topic_matrix:
		topic_list.append([cv.get_feature_names()[j] for j in topic.argsort()[-top_words:]])

# Generate the document topic matrix
	doc_topic_matrix = ctm.get_doc_topic_distribution(dataset)
	doc_topic_list = doc_topic_matrix.argmax(axis=1)

	return topic_list, doc_topic_list


if __name__ == '__main__':
# Define the dataset and the arguments
	df = pd.read_csv(opt.dataset)
	articles = df['content']

# Generate the document term matrix and the vectorizer
	processed_articles = articles.apply(tokenizer)
	cv, dtm = document_term_matrix(processed_articles, opt.vectorizer, opt.min_df, opt.max_df)
# Generate the bag-of-words, the dictionary, and the word2vec model trained on the dataset
	bow, dictionary, w2v = get_dictionary(cv, articles, opt.min_df, opt.size, opt.sg)

# Generate the sBert embeddings
	sent_embeddings = sBert_embeddings(articles, device)

# Create the dataset
	dataset = dataset_creation(dtm, sent_embeddings)

# Generate the list of lists of the top 10 words of each topic and the proportion of topics over the documents
	vocab_size = dtm.shape[1]
	topic_list, doc_topic_list = CTM(dataset, cv, vocab_size, opt.bert_size, opt.num_topics, opt.top_words, opt.epochs)

# Print the title of the document and its topic based on CTM
	df['Topic'] = doc_topic_list
	print(df[['title', 'Topic']])

# Calculate the coherence scores
	evaluation_model = CoherenceScores(topic_list, bow, dictionary, w2v)
	coherence_scores = evaluation_model.get_coherence_scores()
# Print the coherence scores C_V, NPMI, UCI, and C_W2V, respectively
	print(coherence_scores)