from decouple import config
import tweepy
import basilica

# Pytorch and BERT
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
mallet_path = '/Users/mattkirby/Social-Analysis/tweet-analysis/mallet-2.0.8/bin/mallet'

# Spacy for lemmatization
import spacy
nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

# NLTK
import nltk
from nltk.corpus import stopwords


#Preprocess for BERT
def bert_preprocess(list_of_stings):
    sentences = []
    begin_tag = "[CLS] "
    end_tag = " [SEP]"
    for tweet in list_of_stings:
        for sentence in tweet.split('.'):
            sentences.append(begin_tag + ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", sentence).split()) + end_tag)
    indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)) for sentence in sentences]
    return indexed_tokens


# Get BERT embeddings
def BERT_embeddings(list_sentences):
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    for indexed_tokens in list_sentences:
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([0] * len(indexed_tokens))

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()

# Create corpus
def new_corpus(tweets):

    #Tokenize
    # Clean corpus
    def tokenize(corpus):
        for tweet in corpus:
            yield(gensim.utils.simple_preprocess(
                str(tweet.full_text), deacc=True))

    tokens = list(tokenize(tweets))

    # Remove stopwords
    stop_words = stopwords.words('english')
    stop_words.extend([
        'from',
        'subject',
        're',
        'edu',
        'use',
        'https',
        'try',
        'http'])

    no_stop_tokens = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in tokens]

    # Do lemmatization keeping only noun, adj, vb, adv
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    data_lemmatized = lemmatization(no_stop_tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    global id2word
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    # Term Document Frequency
    corpus = [id2word.doc2bow(tweet) for tweet in data_lemmatized]

    return corpus


# mallet_topic_model
def mallet_topics(corpus):
    ldamallet = gensim.models.wrappers.LdaMallet(
        mallet_path,
        corpus=corpus,
        num_topics=20,
        id2word=id2word
        )
    topics = {}
    for i in range(0, ldamallet.num_topics):
        topic = []
        for word_record in ldamallet.print_topic(i).split('+'):
            topic.append((word_record.split("*")[0],
                          word_record.split("*")[1]\
                              .replace('"', "")\
                              .replace(' ', "")))
        topics['topic' + str(i)] = topic
    return topics


#Get embeddings for corpus
def embeddings(tweets):
    tweet_embeddings = {}

    for tweet in tweets:
        # Create embedding
        embedding = BASILICA.embed_sentence(tweet.full_text, model = 'twitter')

    # Create DB record
    tweet_embeddings[str(tweet.id)] = embedding
    return
