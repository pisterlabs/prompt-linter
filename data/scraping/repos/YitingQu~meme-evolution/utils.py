import json
import re, os, sys
import numpy as np
from string import punctuation
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('words')
from nltk.tokenize import word_tokenize
from pyparsing import Keyword
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import pickle
from pathlib import Path
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from tqdm import tqdm
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from googleapiclient import discovery
import requests
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
nltk.download('punkt')

LEXION = set(nltk.corpus.words.words())
STOP_WORDS = set(nltk.corpus.stopwords.words("english"))
PUNCTUATION = set(punctuation)
device = torch.device("cuda")

def clean_text(text):
    """
    input: a list of sentences list
    output: a list of clean splited words
    """
    if isinstance(text[0], list):
        words = []
        for doc in text:
            tokens = [to for line in doc for to in word_tokenize(line)]
            words = [word for word in tokens if ((word.isalpha()) and (len(word)<20))]
            words = [word for word in words if word not in STOP_WORDS]
            words.append(words)
    else:
        tokens = [to for line in text for to in word_tokenize(line)]
        words = [word for word in tokens if ((word.isalpha()) and (len(word)<20))]
        # words = [word for word in tokens if word in LEXION]
        words = [word for word in words if word not in STOP_WORDS]
    return words

def create_wordcloud(text):
    words = clean_text(text)
    wordcloud = WordCloud(width=500, height=500, background_color="white", max_words=1000, contour_width=3, contour_color="steelblue")
    wordcloud.generate(" ".join(words))
    wordcloud.to_image()
    return wordcloud

def extract_keywords_keybert(text, extract_rule, top_n): # check all files with this function
    kw_model = KeyBERT()
    if extract_rule == "ngram":
        keyphrases = kw_model.extract_keywords(docs=text, 
                                                keyphrase_ngram_range=(2, 3),
                                                top_n=top_n,
                                                # stop_words="english",
                                                # use_maxsum=True,)
                                                use_mmr=True,
                                                diversity=0.1)
    elif extract_rule == "vectorize":
        vectorizer = KeyphraseCountVectorizer(workers=-1, pos_pattern='<J.*>*<N.*>')
        keyphrases = kw_model.extract_keywords(docs=text, 
                                                vectorizer=vectorizer,
                                                top_n=top_n,
                                                # stop_words="english",
                                                # use_maxsum=True,)
                                                use_mmr=True,
                                                diversity=0.1)
    keywords = []
    for kw in keyphrases:
        keywords.append(kw[0])
    return keywords

def extract_keywords_tfidf(text, top_n=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(2, 3), stop_words="english")
    tfidf = tfidf_vectorizer.fit_transform(text)
    print(tfidf)


def extract_keywords_rake(text, top_n=3):
    from rake_nltk import Rake
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = []
    ranked_list = r.get_ranked_phrases_with_scores()
    for keyword in ranked_list:
        keywords.append(keyword[1])
        if len(keywords) == top_n:
            break
    return keywords

def extract_keywords_textrank(text, top_n=3):
    import spacy
    import pytextrank

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("textrank")
    doc = nlp(text)
    keywords = []
    for phrase in doc._.phrases:
        if len(phrase.text) > 35:
            continue
        keywords.append(phrase.text)
        if len(keywords) == top_n:
            break
    return keywords

def extract_keywords_yake(text, top_n=3):
    import yake
    kw_extractor = yake.KeywordExtractor()
    keywords_list = kw_extractor.extract_keywords(text)
    keywords = []
    for kw in keywords_list[:top_n]:
        keywords.append(kw[0])
    return keywords

def query_rewire(text):
    api_url = ""
    client_api_key = ""
    response = requests.post(api_url, json={"text": text},headers={"x-api-key": client_api_key})
    res = json.loads(response.text)
    if res["label"] == "non-hateful":
        res["confidence_score"] = 1 - res["confidence_score"]
    return res["confidence_score"]


def query_perspective(text):
    key=""
    client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=key,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
    )

    analyze_request = {
    'comment': { 'text': text },
    'requestedAttributes': {'IDENTITY_ATTACK': {},
                                'SEVERE_TOXICITY': {},
                                'INSULT':{},
                                'SEXUALLY_EXPLICIT':{},
                                'THREAT': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()
    return response["attributeScores"]["IDENTITY_ATTACK"]["summaryScore"]["value"]

class GPT_Perplexity:
    def __init__(self):
        model_id = "gpt2-large"
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    def cal_perp(self, sentence):
        encodings = self.tokenizer(sentence, return_tensors="pt")
        max_length = self.model.config.n_positions # 1024
        stride = 3

        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return ppl.detach().cpu().numpy()


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]



if __name__ == "__main__":
    # create_wordcloud(text)
    # build_topics(text)
    # query_topics(text)

    test_text = ['traditionalism european identity movements share links pdfs reading videos propaganda']
    clean_txt = clean_text(test_text)
    print(clean_txt)

