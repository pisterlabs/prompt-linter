import collections
import itertools
import pickle
import nltk
import spacy
import pandas as pd
import textcleaner as tc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from docutils.nodes import section, colspec
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize,word_tokenize
import gensim
import pandas as pd
import numpy as np
import string
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from random import choice
import os.path
from gensim import corpora
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim.similarities.docsim
import torch
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from Early_Stopping import EarlyStopping
from CNN_encoder import CNN_Encoder

nlp = spacy.load('en_core_web_sm')
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")

lsa_training=False
cnn_training=False
pretrained_word2vec=False
load_representation = False

lsi_model_path = 'models/LSI_model/lsi.model'
corupus_lsi_path = 'models/LSI_model/corpus_lsi.pickle'
dictionary_path = 'models/LSI_model/dictionary.txt'
word2vec_model_path = 'models/LSI_model/Word2Vec_with_new_lemma.model'
cnn_model_path = 'models/LSI_model/Cosine_loss/train_loss-0.25783960384783156_epoch-50_checkpoint.pth.tar'
load_representation_path ='models/LSI_model/representations.pth'
cnn_model_save_directory_path='models/LSI_model/Cosine_loss'

number_of_topics = 200
EMDEDDING_DIM = 100
HID_DIM = 200
OUTPUT_DIM = 1063#number_of_topics
BATCH_SIZE = 16
EPOCH = 100

def sentence_tokenizer(sentences:str):
    raw_sentences = []
    doc = nlp(sentences)
    for i, token in enumerate(doc.sents):
        print('-->Sentence %d: %s' % (i, token.text))
        raw_sentences.append(token.text)
    return raw_sentences

def new_lemmatization(sentences:[str],allowed_postags=['NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN','VBP', 'VBZ', 'JJ', 'JJR', 'JJS']):
    texts_out = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)
        pos_cleaned_sent = " ".join([token for (token, pos) in tagged if pos in allowed_postags])
        doc = nlp(pos_cleaned_sent)
        # Extract the lemma for each token and join
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc]))
    return texts_out

def remove_null_sentence(sentences:[str]):
    return [x for x in sentences if x is not '']


def word_tokenizer(sentences:[str]):

    sentence=[]
    for raw_sentence in sentences:
    # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
        # Otherwise, get a list of words
            sentence.append(word_tokenize(raw_sentence))
    return sentence
    # Return the list of sentences
    # so this returns a list of lists

def text_preprocessing(sentences:[str]):

    input_text = list(tc.document(sentences).remove_numbers().remove_stpwrds().remove_symbols().lower_all())
    lema = new_lemmatization(sentences=input_text)
    return lema

def prepare_corpus(doc_clean,lsa_training=True):
    """
    Input  : clean document
    Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
    Output : term dictionary and Document Term Matrix
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    if lsa_training:
        dictionary = corpora.Dictionary(doc_clean)
        dictionary.save(dictionary_path)
        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    else:
        dictionary = Dictionary.load(dictionary_path)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    return dictionary,doc_term_matrix

def create_gensim_lsa_model(doc_clean,number_of_topics,lsa_training=True):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    if lsa_training:

        dictionary,doc_term_matrix=prepare_corpus(doc_clean,lsa_training)
        # generate LSA model
        lsi_model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        #coherence_value = CoherenceModel(model=lsi_model, texts=doc_clean, dictionary=dictionary, coherence='c_v').get_coherence()
        #print("Coherence value : ",coherence_value)
        print('Saving lsi_model...')
        lsi_model.save(lsi_model_path)
        print('lsi_model saved!')
        corpus_lsi = lsi_model[doc_term_matrix]
        with open(corupus_lsi_path, 'wb') as handle:
            pickle.dump(corpus_lsi, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Corpus_lsi saved.')

    else:

        dictionary,doc_term_matrix=prepare_corpus(doc_clean,lsa_training)
        print('Loading lsi_model...')
        lsi_model=LsiModel.load(lsi_model_path)
        print('lsi_model Loaded!')
        corpus_lsi = lsi_model[doc_term_matrix]

    return lsi_model,corpus_lsi,dictionary

def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=1):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def plot_graph(doc_clean,start, stop, step):
    dictionary,doc_term_matrix=prepare_corpus(doc_clean)
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix,doc_clean,stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

def load_checkpoint(checkpoint_path, model, optimizer):
    """ loads state into model and optimizer and returns:
        epoch, model, optimizer
    """
    #model_path = 'models/seq2seq/without_batchnorm'
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(epoch, checkpoint['epoch']))
        return epoch,model, optimizer
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))


def find_similarity(vec_lsi,df,k=30):

    with open('models/LSI_model/corpus_lsi.pickle', 'rb') as handle:
        corpus_lsi = pickle.load(handle)
    index = MatrixSimilarity(corpus_lsi,num_features=72)
    sims = index[vec_lsi]
    index = sims[0].argsort()[-k:][::-1]
    for i in index:
        print(i,"------->",df[i])
    return index

def target_dataset(path,lsi_model):

    with open(path, 'rb') as handle:
        corpus_lsi = pickle.load(handle)
    # if cnn_training:
    #     index = MatrixSimilarity(corpus_lsi)
    # return index.index
    temp=[]
    for i in corpus_lsi:
        value,index=  torch.tensor([abs(j[1]) for j in i]).max(dim=0)
        temp.append(torch.from_numpy(lsi_model.get_topics()[index.item()])*value.item())
    return torch.stack(temp)

def prepare_dataset(sentence,pretrained_model,Batch_size=16,Max_len=15,cnn_training=True,lsi_model=None):

    preprocessed_sentences = remove_null_sentence(text_preprocessing(sentences=sentence))
    input_sentence = word_tokenizer(preprocessed_sentences)
    if cnn_training:
        #dictionary = Dictionary(input_sentence)
        #dictionary.save("models/LSI_model/word2vec_dict_with_new_lemma.pkl")
        dictionary = Dictionary.load(dictionary_path)

    else:
        #dictionary = Dictionary.load("models/LSI_model/word2vec_dict_with_new_lemma.pkl")
        dictionary = Dictionary.load(dictionary_path)

    sentence_embeddings=[]
    for sentence in input_sentence:
        sent = []
        for word in sentence:
            sent.append(torch.from_numpy(pretrained_model.wv.word_vec(word)).to(device))
        feature_tensor = torch.stack(sent)
        if len(sentence) < Max_len:
            for i in range(Max_len-len(sentence)):
                feature_tensor = torch.cat((feature_tensor,feature_tensor[i].unsqueeze(0)))
        else:
            feature_tensor = feature_tensor[:Max_len]
        sentence_embeddings.append(feature_tensor)

    input_tensor = torch.stack(sentence_embeddings)
    if cnn_training:
        target_tensor = target_dataset(path = corupus_lsi_path,lsi_model=lsi_model).to(device)
        pairs = [(input_tensor[i],target_tensor[i]) for i in range(len(input_sentence))]
    else:
        pairs = [input_tensor]
    train_iterator = torch.utils.data.DataLoader(dataset=pairs, batch_size=Batch_size)
    return train_iterator


def train(model, iterator, optimizer, criterion, epoch_id):
    model.train()
    epoch_loss = 0
    print_interval = 10
    #manhattan_loss = nn.PairwiseDistance(p=1)
    #man_loss = 0
    for idx, batch_xy in enumerate(iterator):
        src = batch_xy[0]
        trg = batch_xy[1].double().to(device)
        optimizer.zero_grad()
        output = model(src)
        output = output.double()
        assert len(src) == len(output)
        assert trg.shape == output.shape
        y = torch.ones(len(src)).double().to(device)
        loss = criterion(output,trg,y)
        #loss = loss.sum() / src.size(0)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if idx % print_interval == 0:
            print('Epoch : {}, sample {}/{} training loss = {}'.format(epoch_id,idx+1,len(iterator),epoch_loss / (idx+1)))
    return epoch_loss / len(iterator)

def generate_embeddings(model, iterator):
    model.eval()
    with torch.no_grad():
        # preprocessed_sentences = remove_null_sentence(text_preprocessing(sentences=sentences))
        # input_sentence = word_tokenizer(preprocessed_sentences)
        # model = model()
        output=[]
        model.p2.register_forward_hook(get_activation('p2'))
        for i,sentence_batch in enumerate(iterator):
            source=sentence_batch[0]
            # print(source.shape)
            # print(sentence)
            # sent = []
            # for word in sentence:
            #     sent.append(torch.from_numpy(Word2vec_model.wv.word_vec(word)).to(device))
            # feature_tensor = torch.stack(sent)
            model(source)
            output.append(activation['p2'].squeeze())
        temp=torch.cat(output)
        # print(temp.shape)
    return temp

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def testing(sentense,representation,model,Word2vec_model,df,processed_sentences,k=30):

    sentences=[]
    sentences.append(sentense)
    input_representation = generate_embeddings(model=model,iterator=prepare_dataset(sentence=sentences,pretrained_model=Word2vec_model,cnn_training=False,Batch_size=1))
    cosine_distance = nn.CosineSimilarity(dim=1)
    manhattan_distance = nn.PairwiseDistance(p=1)
    distance=[]
    #for i in representation:
    output = cosine_distance(input_representation.unsqueeze(0),representation)
    output1 = manhattan_distance(input_representation.unsqueeze(0),representation)
    distance,index = torch.sort(output,descending=True)
    distance1,index1 = torch.sort(output1,descending=True)
    #print(output.shape)
    #distance.append(output.item())
    #distance = np.array(distance)
    #index = distance.argsort()[-k:][::-1]
    for i in index[:k]:
        print(i.item(),"------->",df.iloc[i.item(),:].values[0],"------->",processed_sentences[i.item()],"------->",output[i.item()].item(),"-------->",output1[i.item()].item())

def main():

    #Load Dataset
    df = pd.read_excel('Tickets.xlsx').values.tolist()
    sentence = list(itertools.chain.from_iterable(df))
    preprocessed_sentences = remove_null_sentence(text_preprocessing(sentences=sentence))
    input_sentence = word_tokenizer(preprocessed_sentences)
    # LSA Model
    if lsa_training:

        #start, stop, step = 2, 140, 1
        #plot_graph(input_sentence, start, stop, step)
        lsi_model, corpus_lsi,dictionary = create_gensim_lsa_model(doc_clean=input_sentence,number_of_topics=number_of_topics,lsa_training=True)

    else:

        #while True:
        # temp = []
        # temp.append('RPCG Conflict issue preventing RXCO submittion')
        # preprocessed_sentences = remove_null_sentence(text_preprocessing(sentences=temp))
        # input_sentence = word_tokenizer(preprocessed_sentences)
        lsi_model, test_lsi,dictionary = create_gensim_lsa_model(doc_clean=input_sentence,number_of_topics=number_of_topics,lsa_training=False)
        # index=find_similarity(test_lsi,df)

    model = CNN_Encoder(input_size=EMDEDDING_DIM,hidden_size=HID_DIM,output_size=len(dictionary.token2id)).to(device)
    n_enc_parms = sum([p.numel() for _, p in model.named_parameters() if p.requires_grad == True])
    print(model, n_enc_parms)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CosineEmbeddingLoss()
    early_stopping = EarlyStopping(PATH=cnn_model_save_directory_path, patience=10,verbose=True)

    if pretrained_word2vec:

        Word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)

    else:

        Word2vec_model = Word2Vec(input_sentence, size=100, window=5, min_count=1, workers=8,sg=1)  # replace with bigram_sent,sentence for bigram model, unigram evaluation
        Word2vec_model.train(input_sentence, total_examples=len(preprocessed_sentences), epochs=10,compute_loss=True)
        Word2vec_model.init_sims(replace=True)
        Word2vec_model.save(word2vec_model_path)

    if cnn_training:
         for i in range(EPOCH):
             #for sentence in input_sentence:
             train_iterator = prepare_dataset(sentence=sentence, pretrained_model=Word2vec_model, Batch_size=BATCH_SIZE,Max_len=15,cnn_training=True,lsi_model=lsi_model)
             train_loss = train(model=model, iterator=train_iterator, optimizer=optimizer, criterion=criterion,epoch_id=i)
             early_stopping(model, train_loss, optimizer, i)
             if early_stopping.early_stop:
                 print("Early stopping")
                 break
    else:
         epoch,model,optimizer = load_checkpoint(cnn_model_path,model,optimizer)

    if load_representation:
        Embeddings = torch.load(load_representation_path)
    else:
        Embeddings = generate_embeddings(model=model,iterator=prepare_dataset(sentence=sentence,pretrained_model=Word2vec_model,cnn_training=False,Batch_size=BATCH_SIZE))
        torch.save(Embeddings,load_representation_path)
        print("total {} representation saved".format(len(Embeddings)))
    print("=============== Representation Loaded =============")

    print("================Testing========================")
    df = pd.read_excel('Tickets.xlsx')
    pd.set_option('display.max_columns', 500)
    while True:
        input_ticket = input('Type Ticket text : ')
        # result=
        testing(sentense=input_ticket, representation=Embeddings, model=model, Word2vec_model=Word2vec_model,df=df, processed_sentences=preprocessed_sentences, k=30)
        # print(result.values)

if __name__ == '__main__':
    main()
