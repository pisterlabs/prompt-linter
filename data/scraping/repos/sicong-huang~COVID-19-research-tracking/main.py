import argparse
import xml.etree.ElementTree as ET
import pickle
import os

from tqdm import tqdm
import numpy as np
import scispacy
import spacy
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from article import Article

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_file', type=str, default='mendeley_document_library_2020-03-25.xml',
                        help='path to the XML to read from')
    parser.add_argument('--out_dir', type=str, default='groups', help='output directory')
    parser.add_argument('--num_topics', type=int, default=5, help='number of groups/topics to form')
    parser.add_argument('--force_read', action='store_true', help='force to read from XML file')
    parser.add_argument('--force_train', action='store_true', help='force to trian new model')
    return parser.parse_args()

def read_xml(path):
    '''
    Takes in path to an XML file, return a list of records
    Args: str
    Returns: list(ET.Element)
    '''
    tree = ET.parse(path)
    records = tree.getroot()[0]
    return records

def filter_valid_records(records):
    ''' 
    Takes in a list of ET.Element and returns a list of ones that contains abstract
    '''
    valid_records = []
    for record in records:
        if record.find('./abstract') != None:
            valid_records.append(record)
    return valid_records

def transform_corpus(articles):
    ''' convert to BOW representation + preprocessing '''
    docs = []
    for article in articles:
        docs.append(article.tokens())
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=10)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    return corpus, dictionary

def train_lda(corpus, dictionary, num_topics=5, passes=10):
    ''' returns an LdaModel according to parameters '''
    model = LdaModel(corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
    return model

def topic_distributions(corpus, model):
    distributions = []
    for bow in corpus:
        distri = model[bow]
        distributions.append(distri)
    return distributions

def hist(topic_assignments, num_topics):
    histogram, bin_edges = np.histogram(topic_assignments, bins=np.arange(num_topics + 1))
    return histogram

def to_ndarray(distribution, num_topics):
    ''' Convert list of tuples into np.ndarray '''
    arr = np.zeros((len(distribution), num_topics))
    for i, distri in enumerate(distribution):
        for j, prob in distri:
            arr[i, j] = prob
    return arr

def cos_similarity(distribution):
    dot_product = np.dot(distribution, distribution.T)
    norm = np.linalg.norm(distribution, axis=1)
    norm_product = np.outer(norm, norm)
    return dot_product / norm_product

def degree_centrality(g):
    ''' find the index of the node with highest degree centrality '''
    s = np.sum(g, axis=1)
    return np.argmax(s)

def find_central_articles(distribution, topic_assignments, num_topics):
    ''' returns indices representative articles for each topoic '''
    central_docs = []
    for i in range(num_topics):
        distri = distribution[topic_assignments == i]
        sim = cos_similarity(distri)
        central_idx = degree_centrality(sim)
        mapping = np.argwhere(topic_assignments == i).squeeze()
        central_docs.append(mapping[central_idx])
    return central_docs

def main():
    args = parse_args()
    root_name = args.xml_file.rsplit('.', 1)[0]
    spacy_model = 'en_core_sci_md'
    
    if not os.path.exists('cache'):
        os.mkdir('cache')

    articles = None
    pickle_file = 'cache/articles_{}.pkl'.format(root_name)
    if os.path.exists(pickle_file) and not args.force_read:
        print('loading from {}'.format(pickle_file))
        articles = pickle.load(open(pickle_file, 'rb'))
    else:
        print('reading from {}'.format(args.xml_file))
        records = read_xml(args.xml_file)
        print('found {} articles'.format(len(records)))
        records = filter_valid_records(records)
        nlp = spacy.load(spacy_model)
        articles = [Article(record, nlp) for record in tqdm(records)]
        pickle.dump(articles, open(pickle_file, 'wb'))

    print('using {} valid articles'.format(len(articles)))

    model = None
    model_file = 'cache/model_{}'.format(root_name)
    force_train = False
    corpus, dictionary = transform_corpus(articles)
    if os.path.exists(model_file) and not args.force_train:
        print('loading trained model from {}'.format(model_file))
        model = LdaModel.load(model_file)
    else:
        print('training LDA')
        model = train_lda(corpus, dictionary, num_topics=args.num_topics)
        model.save(model_file)

    print('Finished training/loading')
    cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
    coherence = cm.get_coherence()
    print('Topic coherence:', coherence)
    
    topics = model.show_topics(num_words=5, formatted=False)
    distributions = topic_distributions(corpus, model)
    distributions = to_ndarray(distributions, len(topics))
    topic_assignments = np.argmax(distributions, axis=1)
    histogram = hist(topic_assignments, len(topics))
    central_indices = find_central_articles(distributions, topic_assignments, len(topics))
    print('Num of topics: {}'.format(len(topics)))

    # below is for output
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    output_files = []
    for i in range(len(topics)):
        f_name = os.path.join(args.out_dir, 'group{}.txt'.format(i))
        output_files.append(open(f_name, 'w'))
    
    for i, topic in enumerate(topic_assignments):
        output_files[topic].write('{}\n'.format('-'*50))
        output_files[topic].write('{}\n'.format(str(articles[i])))
        output_files[topic].write('{}\n\n'.format('='*50))

    for f in output_files:
        f.close()

    grouping_file = os.path.join(args.out_dir, 'grouping.txt')
    with open(grouping_file, 'w') as f:
        for i in range(len(topics)):
            f.write('Topic {}: {} articles\n'.format(i, histogram[i]))
            f.write('word: {}\n'.format(str([t[0] for t in topics[i][1]])))
            f.write('Representative article:\n')
            a = articles[central_indices[i]]
            f.write('Title: {}\n'.format(a.title))
            f.write('Authors: {}\n'.format(', '.join(a.authors)))
            f.write('DOI: {}\n'.format(a.doi))
            f.write('\n')

if __name__ == '__main__':
    main()
