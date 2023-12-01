import csv
import ipdb
import logging
from operator import itemgetter
from gensim import corpora, models
from gensim.utils import lemmatize
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# create sample documents
print("Reading input file 'input/audits_with_content.csv'")
with open('input/audits_with_content.csv', 'r') as f:
    reader = csv.reader(f)
    documents = list(reader)

print("Remove documents without body")
all_docs = [doc for doc in documents if len(doc) == 4 and doc[2] != '']
documents = [doc[2] for doc in documents if len(doc) == 4 and doc[2] != '']
doc_count = len(documents)

# list for tokenized documents in loop
texts = []

print("Generating lemmas for each of the documents")
for document in documents:
    # clean and lemmatize document string
    raw = document.lower()
    tokens = lemmatize(raw, stopwords=STOPWORDS)
    texts.append(tokens)

print("Turn our tokenized documents into a id <-> term dictionary")
dictionary = corpora.Dictionary(texts)

print("Convert tokenized documents into a document-term matrix")
corpus = [dictionary.doc2bow(text) for text in texts]

print("Generate LDA model")
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=50)

print("Writting topics to file")
topics_file = open('output/gensim_topics.txt', 'w')
topics_list = ldamodel.print_topics(num_topics=10, num_words=5)
topics_string = ['Topic {}: {}'.format(i, topic) for i, topic in topics_list]
topics_file.write("\n".join(topics_string))
topics_file.close()

print("Writing tagged docs to file")
tagged_documents_file = open('output/tagged_data.txt', 'w')
for index, document in enumerate(documents):
    raw = document.lower()
    doc_tokens = lemmatize(raw, stopwords=STOPWORDS)
    doc_bow = dictionary.doc2bow(doc_tokens)
    result = ldamodel[doc_bow]
    tag = max(result, key=itemgetter(1))[0]
    tag_string = 'Document {} on {}: Tagged with topic {}\n'.format(index+1, all_docs[index][0], str(tag))
    tagged_documents_file.write(tag_string)
tagged_documents_file.close()
