import sys 
import os
sys.path.append("../")
import re 
import json
import core
import openai
import config
import pathlib
import logging
import requests
import itertools  
import numpy as np
import transformers
from typing import List
from bs4 import BeautifulSoup 
from urllib.parse import urlparse, urljoin


def Connect2Seeking (trigger:bool) -> dict:
    """ Connect to SeekAplha
    Check Last News and Trading New
    
    Parameters:
        trigger : bool, True to initiate Connection 
        
    Return 
        tmp_container: dict {'latest-articles':{'article':'lnk'}, 'market-news':{'article':'lnk'}'
        
    """
    if not trigger:
        return {}
        
    # Handle Error 
    try:
        for section in core.Extension:
            
            # Initating Conection 
            with requests.session() as session:

                req = session.get(urljoin(core.UrlBase, section))
                req.raise_for_status()
                bs = BeautifulSoup(req.content, "html.parser")

                # Headers
                tmp_key = list(core.Headers.keys())[0]
                tmp_headers = bs.find(tmp_key, attrs = core.Headers[tmp_key])

                # Creating tmp_container 
                if section not in core.TMP_CONTAINER.keys():
                    core.TMP_CONTAINER[section] = {}

                # Getting Articles
                tmp_key_article = list(core.Target.keys())[0]
                for article in tmp_headers.find_all(tmp_key_article, attrs=core.Target[tmp_key_article]):
                    core.TMP_CONTAINER[section][article.text]= urljoin(core.UrlBase, article['href'])
                    
    except requests.exceptions.HTTPError as errh:
        logging.error("HTTP Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        logging.error("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        logging.error("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        logging.error("Something went wrong:", err)
    return core.TMP_CONTAINER

# =======================================
# Process Document Step 1.
# =======================================

def ProcessingDocuments (string:str)-> List[tuple]:
    """ Preprocess the Whole Article
    Parameters:
        string: Article Selected by User

    Return:
        article, List[(Article Section, Article Text)]
    """
    # Parameters
    Limit = 1000
    end_index = 0
    article = []

    # Preprocessing Tokens
    tmp_string = string # Copy
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2", cache_dir = os.path.join(config.TOKENIZER_DIR, "models--gpt2"))
    tokens = tokenizer.tokenize(tmp_string)
    num_tokens = len(tokens)

    if num_tokens > Limit:
        # Splitting Document 
        for num, section in enumerate(range(0, num_tokens, Limit)):
            end_index += section + Limit
            article.append((f"section_{num+1}" , "".join(tokenizer.convert_tokens_to_string(tokens[section:end_index]))))
    else:
        article.append((f"section_0" , "".join(tokenizer.convert_tokens_to_string(tokens))))

    return article

# =======================================
#  Document Ebeddings Step 2.
# =======================================

def DocEmbedding (corpus:tuple, save_at:str)->dict:
    """ Connect to OpenAI
    Transform page into embeddings
    
    Args:
        corpus: tuple(section, text),
        selected_option: option selected by user
        
    return:
        dict, {'section_n': (article, floats -> Embeddings)}
    """

    # Settings
    EMBEDDINGS = {}
    KEYS_ = core.DOC_KEYS
    EMBEDDING_PARAMS = core.EMBEDDING_PARAMS
    Keep_Server = core.BASE[1] if core.SERVER.lower() == openai.api_type else core.BASE[0]
    EMBEDDING_PARAMS.pop(Keep_Server, None)

    for section, txt in corpus:
        txt = txt.replace("\n", " ")
        EMBEDDING_PARAMS['input'] = txt
        embeddings = openai.Embedding.create(**EMBEDDING_PARAMS)
        EMBEDDINGS[section] = (txt, embeddings[KEYS_[0]][0][KEYS_[1]])


    # Saving Embeddings
    with open(save_at, 'w', encoding='utf-8') as JsonSave:
        json.dump(EMBEDDINGS,JsonSave, ensure_ascii=False)

    return EMBEDDINGS

# =======================================
# Getting Article Step 3.
# =======================================

def GetArticle(Option:str, Selected:str, return_article:bool=False)-> List[tuple]:
    """ Connect to seeking Apha 
    Scrap and Transform Document 
    
    Parameters:
        Option : str, Latest, Trending
        Selected: str, option selected by user
        return_article: True, return Article but does not do the embedding
    Return 
        Article : pre-defined Message, List[(text, TotalTokens)] """

    # Parameters
    Article_ = []
    Limit = 1000

    try:
        tmp_file_name = re.sub("\W+","",Selected)
        save_at = os.path.join(config.EMBEDDINGS_DIR, f"{tmp_file_name}.json")

        # Check if Document was Selected Before
        if not os.path.isfile(save_at):

            # Starting Connection 
            with requests.session() as session:
                req = session.get(core.TMP_CONTAINER[Option][Selected])
                req.raise_for_status()
                bs = BeautifulSoup(req.content, 'html.parser')

                # Article 
                tmp_key_article = list(core.Article.keys())[0]
                tmp_article = bs.find(tmp_key_article, attrs=core.Article[tmp_key_article]).text.strip()
                    
                # Preprocessing Tokens
                tmp_section = ProcessingDocuments(tmp_article)

                # Create Embeddings
                Embedding = DocEmbedding(corpus = tmp_section, save_at=save_at)
                Message_ =  "Loading , This is my First Time Reading this Document"
        else:
            # Load Doc &Embeddings
            with open (save_at, mode= "rb") as JsonFile:
                Embedding= json.load(JsonFile)

            Message_ = "It seems someone already asked me something about this article , Let me check"
    
    except requests.exceptions.HTTPError as errh:
        logging.error("HTTP Error:", errh)
    except requests.exceptions.ConnectionError as errc:
        logging.error("Error Connecting:", errc)
    except requests.exceptions.Timeout as errt:
        logging.error("Timeout Error:", errt)
    except requests.exceptions.RequestException as err:
        logging.error("Something went wrong:", err)


    return Message_, Embedding

# =======================================
# Getting Question Embeddings Step 4.
# =======================================

def GetQuestionEmbedding(Question:str) -> np.array:
    """ Connect to Open AI and Retrieve Query Embedding
    Question : str, question given by user " 
        
    Return 
        np.array which is the QueryEmbedding"""

    DICT_KEYS = core.DOC_KEYS
    QUESTION_PARAMS = core.QUESTION_PARAMS
    Keep_Server = core.BASE[1] if core.SERVER.lower() == openai.api_type else core.BASE[0]
    QUESTION_PARAMS.pop(Keep_Server, None)
    QUESTION_PARAMS['input'] = Question
    
    QuestionEmb = openai.Embedding.create(**QUESTION_PARAMS)
    return np.array(QuestionEmb[DICT_KEYS[0]][0][DICT_KEYS[1]])

# =======================================
# Comparing Question and Article Step 5.
# =======================================

def ComparingQuestion(QuestionEmb, DocEmb) -> list:
    """ Use Cosine Similarity to filter document Embeddings
    Parameters:
        QuestionEmb: Embedding representation from question,
        DocEmb: Embedding representation from Document

    Return
        List with the higest doc Similiraty
        (CosinSimilarity, Article Part, Text)
    """

    DICT_KEYS = core.COMPARING_KEY
    doc_similarity = (sorted([(np.dot(QuestionEmb, np.array(DocEmb[key_section][1])) , DocEmb[key_section][0]) for key_section in DocEmb.keys()],reverse= True))

    return doc_similarity

# =======================================
# Asking Chatbot Step 6.
# =======================================

def SendingQuestion(Question:str, Mode:str, Doc:str)-> str:

    DICT_KEYS = core.ANSWER_KEYS
    Keep_Server = core.BASE[1] if core.SERVER.lower() == openai.api_type else core.BASE[0]
    core.ANSWER_PARAMS.pop(Keep_Server, None)

    # OpenChatModelc
    if Mode == core.OPTIONS_MODE[0]:
        Statement = core.HEADERS[0] + Doc[1] + core.QATOKENS[0] + Question + core.QATOKENS[3] + core.QATOKENS[1] 
        print(Statement)
        core.ANSWER_PARAMS['prompt'] = Statement
        _response = openai.Completion.create(**core.ANSWER_PARAMS)

    elif Mode == core.OPTIONS_MODE[1]:
        Statement = core.HEADERS[1] + Doc + core.QATOKENS[2]
        core.ANSWER_PARAMS['prompt'] = Statement
        _response = openai.Completion.create(**core.ANSWER_PARAMS)

    return _response[DICT_KEYS[0]][0][DICT_KEYS[1]].strip(" \n")


def UserQuestion (Option:str, Article:str, Mode:str='OpenChatbot', Question:str=None)-> str:
    """ Query Question and Initiate Chatbot
    Parameters:
        Question : str, Question given by the user
        Option: str, latest-articles | market-news
        Mode : OpenChatbot : Free Question, Summary : Create Summary only
    """

    # Load The Article 
    Doc, Stop = None , 0
    Message, ArticleSelected = GetArticle(Option,Article)
    
    # OpenChatbot
    if Mode == core.OPTIONS_MODE[0]:
        EmbQuestion = GetQuestionEmbedding(Question)
        Top_Realted = ComparingQuestion(EmbQuestion,ArticleSelected)
        Doc = Top_Realted[0]

    # Summary
    if Mode == core.OPTIONS_MODE[1]:
        Doc = "".join([text for _, (text, score) in itertools.islice(sorted(ArticleSelected.items(), key=lambda x: x[1][0]), 2)])
        
    Answer = SendingQuestion(Question, Mode = Mode, Doc = Doc)
    return Message, Answer