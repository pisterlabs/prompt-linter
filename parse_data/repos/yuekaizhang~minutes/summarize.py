from models.llm import AutoLLM, ChatGLM
from langchain import LLMChain, PromptTemplate
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import BiliBiliLoader
from typing import List, Dict
from utils import chunk_strings

def load_from_transcripts(transcript_list: List[Dict]) -> List[Document]:
    sentence_list = []
    for transcript_dict in transcript_list:
        sentence_list.append(transcript_dict['s'])

    text_splitted, segmenation_indexs = chunk_strings(sentence_list, 1024)
    docs = [Document(page_content=text) for text in text_splitted]
    return docs, segmenation_indexs

def summarize_contents_and_titles(llm, docs, summarization_prompt_template: str, title_prompt_template: str):
    PROMPT = PromptTemplate.from_template(summarization_prompt_template)
    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)
    results = chain({"input_documents": docs}, return_only_outputs=True)
    print(results)
    summarization = results['output_text']
    summarization_all = results['intermediate_steps'] + [summarization]

    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(title_prompt_template))

    titles = []
    for idx, text in enumerate(summarization_all):
        results = llm_chain(text, return_only_outputs=True)
        titles.append(results['text'])

    return summarization_all, titles

if __name__ == '__main__':
    # loader = BiliBiliLoader(
    #     ["https://www.bilibili.com/video/BV1xt411o7Xu/"]
    # )
    # docs = loader.load()
    # print(docs)
    # exit(0)
    with open("/mnt/samsung-t7/yuekai/llm/data/train.1.csv", encoding='utf-8') as f:
        for line in f:
            results = line.split('\t')
            if results[0] == 'idx':
                continue
            data = json.loads(results[1])

            sentences = data['sentences']
            sentence_dict = {}
            setence_list = []
            for sentence in sentences:
                print(sentence, type(sentence))

                sentence_dict[sentence['id']] = sentence['s']
                setence_list.append(sentence['s'])


    # text_splitter = CharacterTextSplitter(        
    #     separator = "\',",
    #     chunk_size = 1024,
    #     chunk_overlap  = 0,
    #     length_function = len,
    # )

    # text = str(sentence_dict)

    # text_splitted = text_splitter.split_text(text)
    # text_splitted = [ensure_brackets(s) for s in text_splitted]

    # text_splitter = CharacterTextSplitter(        
    #     separator = " ",
    #     chunk_size = 1024,
    #     chunk_overlap  = 0,
    #     length_function = len,
    # )

    # text = " ".join(setence_list)

    # text_splitted = text_splitter.split_text(text)
    text_splitted, segmenation_indexs = chunk_strings(setence_list, 1024)
    docs = [Document(page_content=text) for text in text_splitted[:3]]

    #llm = AutoLLM("/mnt/samsung-t7/yuekai/llm/models/openbuddy-7b-v1.1-bf16-enc")
    #llm = ChatGLM("/mnt/samsung-t7/yuekai/llm/models/chatglm-6b")
    llm = ChatOpenAI()

    summarization_prompt_template = """用一句话总结下面的会议:\n\n{text}\n\n 要求：1.非常简短。\n2.不要出现“会议”等字眼。\n总结："""
    PROMPT = PromptTemplate(template=summarization_prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)


    results = chain({"input_documents": docs}, return_only_outputs=True)
    print(results)
    summarization = results['output_text']
    inputs_list = results['intermediate_steps'] + [summarization]

    prompt_template = "为下面文字生成标题:\n{text}\n要求:\n1.不超过十个字。\n2.非常非常简短 \n 标题："
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template))

    titles = []
    for idx, text in enumerate(inputs_list):
        results = llm_chain(text, return_only_outputs=True)
        titles.append(results['text'])
        if idx < len(segmenation_indexs): 
            print(segmenation_indexs[idx], titles[-1])

    print(titles)