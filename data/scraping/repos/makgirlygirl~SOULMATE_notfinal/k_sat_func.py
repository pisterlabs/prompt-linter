import openai
import random
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from wordhoard import Antonyms, Synonyms
import string
from nltk import ne_chunk
from difflib import SequenceMatcher
from keybert import KeyBERT

## preset for openai key
OPENAI_API_KEY = 'your API KEY'
openai.api_key = OPENAI_API_KEY

## preset for gpt3 patameter
ENGINE='text-davinci-003'
MAX_TOKENS = 2000
TEMPERATURE = 1
TOP_P=1

#%%
## function using gpt3 -> sentence generate
def get_paraphrased_sentences_1(passage:str, option=None)->str:
    goal = 'Paraphrase this passage in one sentence with up to 20 words.'
    if option=='Q8': goal = 'Paraphrase this passage in 1 sentence.'

    prompt = goal + '\n\n' + passage

    response=openai.Completion.create(engine=ENGINE,
                          prompt=prompt,
                          max_tokens=MAX_TOKENS,
                          temperature=TEMPERATURE,
                          stream=False)

    paraphrase=response['choices'][0]['text'].strip()
    if paraphrase == '' or paraphrase == None: return None
    return paraphrase

def get_paraphrased_sentences_n(passage:str, num:int)->list:
    goal = f'Make {num} paraphrased sentences with up to 20 words based on this passage without any numbering, wrap the sentence inside [].'
    prompt = goal + '\n\n' + passage
        
    response=openai.Completion.create(engine=ENGINE,
                          prompt=prompt,
                          max_tokens=MAX_TOKENS,
                          temperature=TEMPERATURE,
                          stream=False)
        
    output_str_list = response['choices'][0]['text'].replace('\n', '___').replace('[', '___').replace(']', '___').split('___')
    paraphrased_list=[]
    for sentence in output_str_list:
        if sentence!='' or len(sentence) > 10:
            paraphrased_list.append(sentence.strip())
    if len(paraphrased_list) < num: return None
    return paraphrased_list[:num]

def get_false_sentences_1(passage:str)->str:
    goal = 'Make a false sentence based on this passage with up to 20 words.'
    prompt = goal + '\n\n' + passage

    response=openai.Completion.create(engine=ENGINE,
                          prompt=prompt,
                          max_tokens=MAX_TOKENS,
                          temperature=TEMPERATURE,
                          stream=False)

    paraphrase=response['choices'][0]['text'].strip()
    if paraphrase == '' or paraphrase == None: return None
    return paraphrase

def get_false_sentences_n(passage:str, num: int)->list:
    goal = f'Make {num}  false sentences with up to 20 words based on this passage without any numbering, wrap the sentence inside [].'
    prompt = goal + '\n\n' + passage
        
    response=openai.Completion.create(engine=ENGINE,
                          prompt=prompt,
                          max_tokens=MAX_TOKENS,
                          temperature=TEMPERATURE,
                          stream=False)
        
    output_str_list = response['choices'][0]['text'].replace('\n', '___').replace('[', '___').replace(']', '___').split('___')
    distractors_list=[]
    for sentence in output_str_list:
        if len(sentence) > 10:
            distractors_list.append(sentence.strip())
    if len(distractors_list) < num: return None
    return distractors_list[:num]

#%%
def check_punctuation_capital_sentence(sentence:str)->str:
    if len(sentence) > 10:
        output=sentence.strip()
        punctuation_marks=['.', '!', '?']
        if output[-1] not in punctuation_marks: 
            output=output+'.'
        if output[0] in punctuation_marks: output=output[1:]
        if output[0].islower(): output=output[0].upper()+output[1:]
        return output.strip()
    return sentence.strip()
    
def get_kwd_n_list(passage: str, top_n:int,  max_word_cnt=1):
    '''
    input: passage(str), top_n:int
    output: kwd_list(list), flag(boolean) 
    '''
    
    flag=True
    kw_model = KeyBERT()
    kwd=kw_model.extract_keywords(passage, keyphrase_ngram_range=(1, max_word_cnt), stop_words='english', top_n=top_n)
    if len(list(kwd)) == 0 : return None

    for i in range(len(kwd)):   ## 점수 빼고 단어만
        kwd[i]=kwd[i][0]

    if len(kwd) < top_n: flag=False

    return kwd, flag

def pos_tagger(nltk_tag):
    '''
    get_pos() 내부에서 쓰임
    '''
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None

def get_pos(word:str)->str:
    tagged_list = pos_tag([word])

    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), tagged_list))
    pos=wordnet_tagged[0][-1]

    return pos

def get_lmtzr(word:str)->str:
    lmtzr = WordNetLemmatizer()
    pos=get_pos(word)
    if pos!=None:
        lmtzr=lmtzr.lemmatize(word, pos)
    else:
        lmtzr=lmtzr.lemmatize(word)
    return lmtzr

def del_same_lemmatization(word_list:list)->list:
    if type(word_list)!=list or len(word_list) == 0:return None
    lmtzr = WordNetLemmatizer()
    word_list_lemmatize=[]
    for i in word_list:
        pos=get_pos(i)
        if pos!=None:
            word_list_lemmatize.append(lmtzr.lemmatize(i, pos))
        else:
            word_list_lemmatize.append(lmtzr.lemmatize(i))
    if len(word_list_lemmatize)==0: return None
    return list(dict.fromkeys(word_list_lemmatize))

def del_same_start(word_list:list, num_spell=4)->list:
    if type(word_list)!=list or len(word_list) == 0: return None
    output=[]
    start_spell_dict=dict()
    random.shuffle(word_list)

    for i in word_list:
        if len(i)>= num_spell:
            start_spell=i[:num_spell]
        else:
            start_spell=i
        start_spell_dict[start_spell]=i

    output=list(start_spell_dict.values())
    if len(output)==0 : return None
    return output

def get_synonym_list_gpt(word:str, num: int)->list:
    goal = f'find {num}  synonym of this word without any numbering, wrap each word inside [].'
    prompt = goal + '\n\n' + word
    flag = True
        
    response=openai.Completion.create(engine=ENGINE,
                          prompt=prompt,
                          max_tokens=MAX_TOKENS,
                          temperature=TEMPERATURE,
                          stream=False)
        
    output_str_list = response['choices'][0]['text'].replace(',', '___').replace('[', '___').replace(']', '___').split('___')
    distractors_list=[]
    for word in output_str_list:
        if len(word) > 1:
            if word!='' and word[0].isalpha():
                distractors_list.append(word.strip().capitalize())
    if len(distractors_list) == 0 : return None
    if len(distractors_list) < num: flag = False
    return distractors_list[:num], flag

def get_antonym_list_gpt(word:str, num: int)->list:
    goal = f'find {num}  antonym of this word without any numbering, wrap each word inside [].'
    prompt = goal + '\n\n' + word
    flag=True
    response=openai.Completion.create(engine=ENGINE,
                          prompt=prompt,
                          max_tokens=MAX_TOKENS,
                          temperature=TEMPERATURE,
                          stream=False)
        
    output_str_list = response['choices'][0]['text']
    output_str_list=output_str_list.replace(',', '___').replace('[', '___').replace(']', '___').split('___')
    distractors_list=[]
    for word in output_str_list:
        if len(word) > 1:
            if word!='' and word[0].isalpha():
                distractors_list.append(word.strip().capitalize())
    if len(distractors_list) == 0 : return None
    if len(distractors_list) < num: flag=False
    return distractors_list[:num], flag

def get_keyword_list_old(passage, max_word_cnt:int,top_n:int)->list:
    kw_model = KeyBERT()
    kwd_list=kw_model.extract_keywords(passage, keyphrase_ngram_range=(1, max_word_cnt), stop_words='english', top_n=top_n)
    for i in range(len(kwd)):   ## 점수 빼고 단어만
            kwd[i]=kwd[i][0]
    return kwd_list
