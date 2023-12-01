from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
from langchain.document_loaders import TextLoader
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os 
from langchain.callbacks import get_openai_callback
from tqdm import tqdm

from langchain.document_loaders import DirectoryLoader

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain import PromptTemplate, OpenAI, LLMChain

from langchain.llms import OpenAI
import numpy as np

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import re

# spacy.prefer_gpu()
# nlp = spacy.load("en_core_web_md")  # make sure to use larger package!
os.environ["OPENAI_API_KEY"] = ' '

from tool import *
from llm_planner import *
from sentence_similarity import *


def init_db(data_dir, chunk_sizes = [180,150,300]):
    """
    init the Chroma for different's length of text and documents
    args:
        data_dir: string for the directory of the dataset
        chunk_sizes: list of int for the chunk size of different dataset
    
    """
    dataset_db = {}
    dataset_text = {}
    for index, data_set in enumerate( ['vision_data','speech_data','summary_data' ]):
    # for index, data_set in enumerate( ['speech_data' ]):

        embeddings = OpenAIEmbeddings()
        # loader = TextLoader('summary_text.txt', encoding='utf8')
        loader = DirectoryLoader(os.path.join(data_dir, data_set), glob="**/*.txt")
        # loader = DirectoryLoader('speech_dir', glob="**/*.txt")
        documents = loader.load()
        chunk_size = chunk_sizes[index]
        # print(chunk_size, data_set)
        separator = '\n'  if data_set == 'speech_data' else 'text end.'
        text_splitter = CharacterTextSplitter(separator = separator, chunk_size=chunk_size,chunk_overlap=0)
        # text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        db = Chroma.from_documents(texts, embeddings)
        dataset_db[data_set] = db
        if data_set == 'speech_data':
            sorted_frames_text = sorted([i.page_content[3:]  for i in texts], key=lambda x: int(x.split()[1]))
        else:
            sorted_frames_text = sorted([i.page_content  for i in texts], key=lambda x: int(x.split()[1]))
        #TODO: make this more general, for now just for speech At frame
        dataset_text[data_set] = sorted_frames_text
        # print(dataset_db['speech_data'])
        # dataset_text[data_set] = ' '.join([i.page_content for i in texts])
    return dataset_db,dataset_text

def recheck(question, tobe_decided_indexs, dataset_text_list, dataset,updated_interval):
   # the second time filter the result
        # re examine the to be decided intervals

        decided_indexs = []
        for i in tobe_decided_indexs:
            start = i-1
            end = i+2
            start = max(0, start)
            end = min(len(dataset_text_list['summary_data']), end)
            decided_indexs.append(llm_doule_check_relevent_text(question, dataset_text_list['summary_data'][start:end]))
            
        # get the new decided intervals
        new_decided_indexs = [x for i, x in enumerate(tobe_decided_indexs) if decided_indexs[i] == True]
        # print(new_decided_indexs)
        redecided_text_list = []
        for i in new_decided_indexs:
            redecided_text_list.append(dataset_text_list['summary_data'][i])
            
        # extract the to be decided intervals
        re_ranges = extract_number_ranges(redecided_text_list, len(redecided_text_list), 0)
        updated_interval.extend(re_ranges)
        return updated_interval


def question2interval(question,dataset_db,dataset_text_list):
    """
    pipeline for question to interval
        1.find the best dataset for the question depending on the question type
        2.find the similar words of the question
        3.find the k most similar text of the question
        4.check the relevent text of the question with langchain similarity search
        5.recheck the relevent text of the question with OpenAI GPT-3.5
        6.filter the intervals with the relevent text
        7.make the intervals continuous
        8.format the intervals to the format of the frontend
    args:
        question: string for asking question related to video
        dataset_db: dict of database for different dataset{vision_data, speech_data, summary_data}
        dataset_text_list: dict of text list for different dataset{vision_data, speech_data, summary_data}
        
    return:
        final_intervals: list of dict of start and end frame index related to the question
    """
    dataset = llm_choose_database(question)

    similar_word_length = 4 if dataset == 'speech_data' else 10
    similar_question_list = llm_similar_words(question = question, words_number= similar_word_length)
    k = count_target_elements(dataset_text_list[dataset], similar_question_list)
    k = 10 if k <= 10 else min(k, len(dataset_text_list[dataset]))
    k = min(k, 4 )
    print('searching in ', dataset ,'counted k', k, len(dataset_text_list[dataset]))

    docs = dataset_db[dataset].similarity_search_with_score(', '.join(similar_question_list),k =k, filter=None)
    question_relavent_text_list = [i[0].dict()['page_content'] for i in docs]


    if k < 12:

        coarse_result = llm_check_relevent_text(' '.join(similar_question_list), ' '.join(question_relavent_text_list))
        length = 0
    else:
        # when video has a big number of same scenes
        length = 5
        start, end = len(question_relavent_text_list) // 2, len(question_relavent_text_list) // 2 + length + 1
        frame_index_result = recursive_check(' '.join(similar_question_list), question_relavent_text_list, length, start, end)
        coarse_result = extract_number_ranges(question_relavent_text_list, frame_index_result,length )
   
    if coarse_result == []: return None
    #the first time filter the result
    updated_interval, to_be_decided = frame_interval_filter(coarse_result)


    if to_be_decided != []:
        
        tobe_decided_indexs = find_to_be_decided_indices(to_be_decided, dataset_text_list['summary_data'])
        updated_interval = recheck( ' '.join(similar_question_list),  tobe_decided_indexs, dataset_text_list, dataset,updated_interval)
        

        
    updated_interval,_ = frame_interval_filter(updated_interval)
    final_intervals = make_continuous_intervals(updated_interval, padding = 5)
    final_intervals = format_intervals(final_intervals)
    return final_intervals

