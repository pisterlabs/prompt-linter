from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.corpus import brown
import numpy as np
import pandas as pd
import nltk
import spacy
from sense2vec import Sense2Vec
import os
import json
import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from nltk.tokenize import sent_tokenize
from similarity.normalized_levenshtein import NormalizedLevenshtein
from langchain import OpenAI, LLMChain, PromptTemplate

try:
    # Check if NLTK data is already downloaded
    if not nltk.data.find('corpora/brown'):
        nltk.download('brown')
    if not nltk.data.find('corpora/stopwords'):
        nltk.download('stopwords')
    if not nltk.data.find('corpora/words'):
        nltk.download('words')
except:
    print("===== ERROR OCCURRED =====")
finally:
    print("===== NLTK DOWNLOADED =====")
from ques.mcq import tokenize_sentences,get_keywords,get_sentences_for_keyword,generate_questions_mcq


class QGen:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xxl')
        model = T5ForConditionalGeneration.from_pretrained('Parth/result')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.device('cpu'):
            device = torch.device('cpu')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        model.to(device)
        self.device = device
        self.model = model
        try:
            self.nlp = spacy.load('en_core_web_trf')
        except:
            os.system('python -m spacy download en_core_web_trf')
            self.nlp = spacy.load('en_core_web_trf')
        self.s2v = None

        try:
            self.s2v = Sense2Vec().from_disk('s2v_old')
        except:
            try:
                if os.path.exists('s2v_reddit_2015_md'):
                    print("s2v_reddit_2015_md already exists")
                else:
                    print("Downloading s2v_reddit_2015_md")
                    os.system('wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz')
                    os.system('cat s2v_reddit_2015_md.tar.gz* | tar -xzv')
                if os.path.exists('s2v_reddit_2015_md.tar.gz'):
                    print("Extracting s2v_reddit_2015_md")
                    os.system('cat s2v_reddit_2015_md.tar.gz* | tar -xzv')
            except:
                print("Failed to download and extract s2v_reddit_2015_md")
            finally:
                if self.s2v is None:
                    self.s2v = Sense2Vec().from_disk('s2v_reddit_2015_md')

        self.fdist = FreqDist(brown.words())
        self.normalized_levenshtein = NormalizedLevenshtein()
        self.set_seed(42)

    def set_seed(self,seed):
        np.random.seed(seed)
        torch.manual_seed(seed) 
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def predict_mcq(self, payload):
        start = time.time()
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions")
        }

        text = inp['input_text']
        sentences = tokenize_sentences(text)
        joiner = " "
        modified_text = joiner.join(sentences)


        keywords = get_keywords(self.nlp,modified_text,inp['max_questions'],self.s2v,self.fdist,self.normalized_levenshtein,len(sentences) )


        keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)

        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet

   
        final_output = {}

        if len(keyword_sentence_mapping.keys()) == 0:
            return final_output
        else:
            try:
                generated_questions = generate_questions_mcq(keyword_sentence_mapping,self.device,self.tokenizer,self.model,self.s2v,self.normalized_levenshtein)

            except:
                return final_output
            end = time.time()

            final_output["statement"] = modified_text
            final_output["questions"] = generated_questions["questions"]
            # final_output["time_taken"] = end-start

            
            if torch.device=='cuda':
                torch.cuda.empty_cache()
                
            return final_output

def total_words(num_questions,difficulty_level):
    if difficulty_level == "easy":
        total_words = (num_questions * 50 ) + 100
    elif difficulty_level == "medium":
        total_words = (num_questions * 100 ) + 200
    elif difficulty_level == "hard":
        total_words =  (num_questions * 50 ) + 100
    return total_words

if __name__ == '__main__':
    qgen = QGen()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    x = wikipedia.run("python developer")
    # x = MCQGen(8, "easy", "python developer")
    payload = {
            "input_text": x
        }
    # JSON format
    print(json.dumps(qgen.predict_mcq(payload), indent=4))
