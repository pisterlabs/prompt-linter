# From
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

import os
import csv
import argparse

import spacy
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, Phrases
from gensim.models.phrases import Phraser
from gensim.models.ldamodel import LdaModel


def preprocess(sentences, spacy_model, stopwords, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    dataset = []

    for sentence in sentences:
        words = [token.lemma_ for token in spacy_model(sentence) if token.pos_ in allowed_postags]
        words = simple_preprocess(' '.join(words), deacc=True)
        words = [word for word in words if not word in stopwords]
        dataset.append(words)

    # Build the bigram and trigram models
    bigram = Phrases(dataset, min_count=5, threshold=100) # higher threshold fewer phrases.
    # trigram = Phrases(bigram[words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = Phraser(bigram)
    # trigram_mod = Phraser(trigram)

    bigrams = [bigram_mod[sentence] for sentence in dataset]
    return bigrams

def parse_options():
    parser = argparse.ArgumentParser(description='Run Gensim LDA model')
    parser.add_argument('--data', required=True, help='the TSV file that contains the training sentences')
    parser.add_argument('--num-topics', type=int, required=True, help='the number of topics')
    parser.add_argument('--iterations', type=int, default=10, help='number of iterations')
    parser.add_argument('--stopwords-dir', default=os.path.join('resources', 'stopwords'), help='the directory with the stopwords')
    parser.add_argument('--lang', required=True, help='the language of the dataset')
    parser.add_argument('--model-name', required=True, help='the name of the model to generate')
    parser.add_argument('--model-dir', default=os.path.join('models', 'gensim', 'lda'), help='the directory where to store the models')
    return parser.parse_args()

if __name__ == '__main__':
    options = parse_options()

    with open(os.path.join(options.stopwords_dir, f'{options.lang}.txt')) as f:
        stopwords = set([line.strip() for line in f.readlines()])

    with open(options.data) as f:
        reader = csv.reader(f, delimiter='\t')
        data = [line[0] for line in reader]

    model_dir = os.path.join(options.model_dir, options.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model')

    # Initialize spacy model, keeping only tagger component (for efficiency)
    spacy_model = spacy.load(options.lang, disable=['parser', 'ner'])
    print('Preprocessing dataset...')
    texts = preprocess(data, spacy_model, stopwords)
    print('...done')

    # Create Dictionary
    print('Creating dictionary...')
    id2word = corpora.Dictionary(texts)
    print('...done')
    # Term Document Frequency
    print('Creating corpus...')
    corpus = [id2word.doc2bow(text) for text in texts]
    print('...done')

    # Build LDA model
    print('Training LDA model...')
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=options.num_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=options.iterations,
        alpha='auto',
        per_word_topics=True
    )
    print('...done')

    print('Saving model...')
    model.save(model_path)
    print('...done')

    print('Topics found:')
    for i in range(options.num_topics):
        print(i, ' -> ', model.print_topic(i))
    doc_lda = model[corpus]

    # Compute Perplexity
    print('Perplexity: ', model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda)