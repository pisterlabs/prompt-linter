''' Train the model.'''

import argparse
import logging
import os
import re

import nltk
from gensim.corpora import Dictionary
from gensim.models import LdaModel, Phrases
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from azureml.core import Dataset, Run


def get_data(dataset_id):
    '''Get the dataset and perform general preparation.'''
    run = Run.get_context()
    ws = run.experiment.workspace # pylint: disable=invalid-name
    dataset = Dataset.get_by_id(ws, id=dataset_id)
    df = dataset.to_pandas_dataframe() # pylint: disable=invalid-name
    # Keep only text content
    train_docs = df.iloc[:, 2].values
    # Remove empty elements (?)
    train_docs = [d for d in train_docs if d]
    return train_docs


def preprocess_doc(doc):
    '''Preprocess a document for training.'''

    # Remove some stuff before tokenization.

    # Remove email addresses.
    doc = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '', doc)

    # Tokenize the document.

    # Split the document into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    doc = doc.lower()  # Convert to lowercase.
    doc = tokenizer.tokenize(doc) # Split into words.

    # Remove numbers, but not words that contain numbers.
    doc = [token for token in doc if token.isalpha()]

    # Remove words that are only one character.
    doc = [token for token in doc if len(token) > 1]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words.union(['one', 'ax', 'max'])
    doc = [token for token in doc if not token in stop_words]

    # Lemmatize the documents.
    lemmatizer = WordNetLemmatizer()
    doc = [lemmatizer.lemmatize(token) for token in doc]

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(doc, min_count=20)
    for token in bigram[doc]:
        if '_' in token:
            # Token is a bigram, add to document.
            doc.append(token)

    return doc


def preprocess_docs(docs):
    '''Preprocess all the documents and create dict + corpus.'''

    # Pre-process the documents.

    docs = [preprocess_doc(doc) for doc in docs]

    # Remove rare and common tokens.

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # Remove 5 most frequent
    dictionary.filter_n_most_frequent(5)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    return corpus, docs, dictionary


def train_model(corpus, dictionary, args):
    '''Train the LDA model.'''

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        chunksize=args.chunksize,
        alpha='auto',
        eta='auto',
        iterations=args.iterations,
        num_topics=args.num_topics,
        passes=args.passes,
        eval_every=None,
        random_state=314159
    )

    return model


def main():
    '''Main function.'''

    # Set up logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

    # Download NLTK components
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, required=True)
    parser.add_argument('--num-topics', type=int, required=True)
    parser.add_argument('--chunksize', type=int, required=True)
    parser.add_argument('--passes', type=int, required=True)
    parser.add_argument('--iterations', type=int, required=True)
    args = parser.parse_args()

    # Prepare data

    train_docs = get_data(args.input_data)

    corpus, docs, dictionary = preprocess_docs(train_docs)

    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    # Train model
    model = train_model(corpus, dictionary, args)

    # Compute and log Coherence Score
    coherence_model_lda = CoherenceModel(
        model=model,
        texts=docs,
        dictionary=dictionary,
        coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda)

    run = Run.get_context()
    run.log(name='c_v', value=float(coherence_lda))

    # Save model & dictionary
    os.makedirs('outputs', exist_ok=True)
    model.save('./outputs/model.pickle')
    dictionary.save('./outputs/dictionary.pickle')


if __name__ == "__main__":
    main()
