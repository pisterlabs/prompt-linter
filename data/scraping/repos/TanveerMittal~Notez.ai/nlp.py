import re
import gensim
import pandas as pd
import numpy as np
import nltk
# Download NLTK modules
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.utils import simple_preprocess
from collections import Counter, defaultdict
from transformers import pipeline
import math
import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
from question_generation.pipelines import pipeline as qg_pipeline
import streamlit as st

# Load stop words
stop_words = stopwords.words('english')

model = None
tokenizer = None
qg = None
summarizer = None


#@st.cache(hash_funcs={OpenAIGPTModel: id, OpenAIGPTTokenizer: id, qg_pipeline: id, pipeline:id})
def init_models():
    global model, tokenizer, qg, summarizer, stop_words    
    # Load pre-trained model (weights)
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt').eval()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    # Load question generation model
    qg = qg_pipeline("question-generation", use_cuda=True)

    # Load HuggingFace Summarizer
    summarizer = pipeline("summarization")

    return True


def nlp_pipeline(doc, n_topics, n_questions):
    '''Run entire NLP pipeline on zoom transcript'''
    #Initialize models
    if model is None or tokenizer is None or qg is None or summarizer is None:
        init_models()

    # Filter sentences 
    filtered_sents = get_filtered_sents(clean(doc))

    # Convert sentences to lists of words
    data_words = list(sent_to_words(filtered_sents))

    # Remove stop words
    data_words = remove_stopwords(data_words)

    # Construct model vocabulary
    id2word = corpora.Dictionary(data_words)
    
    # Create Corpus
    texts = data_words

    # Compute Term Document Frequency on doc
    corpus = [id2word.doc2bow(text) for text in texts]

    # Fit LDA model to input corpus
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=n_topics)

    # Initalize topic clusters of sentences
    clusters = defaultdict(str)

    # Use LDA to assign topic and confidence to each sentence
    labels = [np.array(lda_model.get_document_topics(sent))[:, 1].argmax() for sent in corpus]
    confidences = [np.array(lda_model.get_document_topics(sent))[labels[i],1] for i, sent in enumerate(corpus)]
    
    # Populate clusters with labeled sentences
    for i in range(len(filtered_sents)):
        clusters[labels[i]] += filtered_sents[i] + " "

    # Clean extra whitespace and punctuation from text
    for topic in clusters:
        clusters[topic] = re.sub(r"\s+", " ", clusters[topic])
        clusters[topic] = re.sub(r"[',]", "", clusters[topic])

    # Build dataframe to get questions for top k sentences for each topic
    df = pd.DataFrame().assign(topic=labels, confidence=confidences, sentences=filtered_sents)
    top_sentences = df.groupby("topic").apply(lambda grp: df.loc[grp['confidence'].nlargest(n_questions).index])
    top_sentences = top_sentences.assign(questions=top_sentences["sentences"].apply(lambda sent: get_most_coherent_qa(qg(sent))))
    top_sentences.index = top_sentences.index.droplevel(0)
    top_sentences = top_sentences.reset_index(drop=True)

    # Build dataframe to get summaries for top 10 sentences from each topic
    summary_sentences = df.groupby("topic").apply(lambda grp: df.loc[grp['confidence'].nlargest(10).index])
    summary_sentences.index = summary_sentences.index.droplevel(0)
    summary_sentences = summary_sentences.reset_index(drop=True)
    to_summarize = summary_sentences.groupby("topic")["sentences"].sum()
    summaries = to_summarize.apply(lambda x: ' '.join([t["summary_text"] for t in summarizer(x)])) 
    
    # Format final output
    output = {}
    for i in range(n_topics):
        output[i] = {"questions":top_sentences[top_sentences["topic"] == i]["questions"].to_list(),
                    "summaries": summaries[i]}
    
    return output

def clean(doc):
    '''Remove punctuation and redundant whitespace'''
    return re.sub(r"[',]", "", re.sub(r"\s+", " ", doc))

def get_teacher_text(doc):
    '''Extracts text spoken by teacher'''
    matches = re.findall(r"(?:(?:[A-Z][a-z]*\s)*(?:[A-Z][a-z]*)):", doc)
    splits = re.split(r"(?:(?:[A-Z][a-z]*\s)*(?:[A-Z][a-z]*)):", doc)[1:]
    teacher = Counter(matches).most_common(1)[0][0]
    return ''.join([splits[i] for i in range(len(splits)) if matches[i] == teacher])

def get_sents(doc):
    return [sent for sent in sent_tokenize(doc)]

def get_filtered_sents(doc):
    '''Split doc into list of sentences and filter small sentences out'''
    return [sent for sent in sent_tokenize(doc) if len(re.findall(r"\s", sent)) > 10]

def sent_to_words(sentences):
    '''Convert sentences to lists of words'''
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    '''Remove stop_words from input texts'''
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def get_most_coherent_qa(qa):
    '''Takes list of questions/answers and finds most coherent pair with GPT-2'''
    if len(qa) == 1:
        return qa[0]
    scores = [score(pair["question"] + " " + pair["answer"]) for pair in qa]
    idx = np.argmax(scores)
    return qa[idx]

def score(sentence):
    '''Calculate Sentence Perplexity with GPT-2'''
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss=model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)

def filter_vtt(data):
    names_count = dict()
    names_lines = dict()
    data = data.split("\n")
    data.pop(0)
    data.pop(0)
    try:
        while True:
            data.remove("")
    except:
        pass
    count=0
    for i in range(len(data)):
        if count%3==0 or count%3==1:
            data.pop(count//3)
        if count%3==2:
            data[count//3]=data[count//3].split(":")
            if len(data[count//3]) == 1:
                data[count//3]=""
            else:
                if data[count//3][0] not in names_count:
                    names_count[data[count//3][0]] = 0
                    names_lines[data[count//3][0]] = list()
                names_count[data[count//3][0]] += 1
                names_lines[data[count//3][0]].append(count//3)
                data[count//3]=data[count//3][1][1:]
        count+=1
    lecturer = None
    max = 0
    for n,c in names_count.items():
        if c > max:
            lecturer=n
            max=c

    to_remove=list()
    for n,l in names_lines.items():
        if n != lecturer:
            to_remove.extend(l)
    to_remove.sort(reverse=True)

    for i in to_remove:
        data.pop(i)

    try:
        while True:
            data.remove("")
    except:
        pass

    return " ".join(data)
    
if __name__ == "__main__":
    with open("econ3_1.txt") as f:
        doc1 = f.read().strip()
    print(get_teacher_text(doc1))