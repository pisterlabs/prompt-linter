import os
import configparser

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from langchain.prompts import PromptTemplate
from reader import _find_answer, _find_sources, _docs_to_string, remove_brackets

import cfg
import json
import tiktoken

config = configparser.ConfigParser()
config.read('apikey.ini')
chat_api_list = config.get('OpenAI', 'OPENAI_API_KEYS')[1:-1].replace('\'', '').split(',')
os.environ["OPENAI_API_KEY"] = chat_api_list[0]

TOP_K = cfg.retriever_top_k
PROMPTS = cfg.prompts
QUERIES = cfg.retrieval_queries
TCFD_ASSESSMENT = cfg.tcfd_assessment
TCFD_GUIDELINES = cfg.tcfd_guidelines
SYSTEM_PROMPT = cfg.system_prompt


class UserQA:
    def __init__(self, llm_name='gpt-3.5-turbo', answer_key_name='ANSWER', max_token=512,
                 root_path='./',
                 gitee_key='',
                 user_name='defualt', language='en'):
        self.user_name = user_name  # user name
        self.language = language
        self.root_path = root_path
        self.max_token = max_token
        self.llm_name = llm_name
        #
        self.tiktoken_encoder = tiktoken.encoding_for_model(self.llm_name)
        self.cur_api = 0
        self.prompts = PROMPTS
        self.answer_key_name = answer_key_name
        self.basic_info_answers = []
        self.user_questions = []
        self.user_answers = []

    def user_qa(self, question, report, basic_info_path, answer_length=60, prompt_template=None,
                top_k=20):
        if prompt_template is None:
            prompt_template = self.prompts['user_qa_source']
        # to_question_prompt = PromptTemplate(template=self.prompts['to_question'], input_variables=["statement"])
        # to_question_message = [
        #     SystemMessage(content="You are a helpful AI assistant."),
        #     HumanMessage(content=to_question_prompt.format(statement=question))
        # ]
        # llm = ChatOpenAI(temperature=0)
        # question = llm(to_question_message).content
        if os.path.exists(basic_info_path):
            with open(basic_info_path, 'r') as f:
                basic_info_dict = json.load(f)
            basic_info_string = str(basic_info_dict)
        else:
            basic_info_prompt = PromptTemplate(template=self.prompts['general'], input_variables=["context"])
            if "turbo" in self.llm_name:
                # title = "Title: " + report.title + '\n'
                # first_page = "First Page: " + report.pdf[0].get_text() + '\n'
                message = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=basic_info_prompt.format(
                        context=_docs_to_string(report.section_text_dict['general'], with_source=False)))
                ]
                llm = ChatOpenAI(temperature=0, max_tokens=256)
                output_text = llm(message).content
            else:
                message = basic_info_prompt.format(
                    context=_docs_to_string(report.section_text_dict['general'], with_source=False))
                llm = OpenAI(temperature=0, max_tokens=256)
                output_text = llm(message)
            try:
                basic_info_dict = json.loads(output_text)
            except ValueError as e:
                basic_info_dict = {'COMPANY_NAME': _find_answer(output_text, name='COMPANY_NAME'),
                                   'COMPANY_SECTOR': _find_answer(output_text, name='COMPANY_SECTOR'),
                                   'COMPANY_LOCATION': _find_answer(output_text, name='COMPANY_LOCATION')}
            basic_info_string = str(basic_info_dict)
            with open(basic_info_path, 'w') as f:
                json.dump(basic_info_dict, f)
            self.basic_info_answers.append(basic_info_dict)

        self.user_questions.append(question)
        # get the retriever, where the vector database is loaded from report.db_path
        retriever, _ = report._get_retriever(report.db_path)
        docs = retriever.get_relevant_documents(question)
        tcfd_prompt = PromptTemplate(template=prompt_template,
                                     input_variables=["basic_info", "summaries", "question", "answer_length"])
        num_docs = top_k
        current_prompt = tcfd_prompt.format(basic_info=basic_info_string,
                                            summaries=_docs_to_string(docs),
                                            question=question,
                                            answer_length=str(answer_length))
        if '16k' not in self.llm_name:
            while len(self.tiktoken_encoder.encode(current_prompt)) > 3500 and num_docs > 10:
                num_docs -= 1
                current_prompt = tcfd_prompt.format(basic_info=self.basic_info_answers[0],
                                                    summaries=_docs_to_string(docs, num_docs=num_docs),
                                                    question=question,
                                                    answer_length=str(answer_length))
        if "turbo" in self.llm_name:
            message = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=current_prompt)
            ]
            llm = ChatOpenAI(temperature=0, max_tokens=512)
            output_text = llm(message).content
        else:
            message = current_prompt
            llm = OpenAI(temperature=0, max_tokens=512)
            output_text = llm(message)
        try:
            answer_dict = json.loads(output_text)
        except ValueError as e:
            answer_dict = {self.answer_key_name: _find_answer(output_text, name=self.answer_key_name),
                           'SOURCES': _find_sources(output_text)}
        page_source = []
        for s in answer_dict['SOURCES']:
            try:
                page_source.append(report.page_idx[s])
            except Exception as e:
                pass
        used_chunks = []
        for doc in docs:
            if int(doc.metadata['source']) in answer_dict['SOURCES']:
                used_chunks.append(doc.page_content)
        answer_dict[self.answer_key_name] = remove_brackets(answer_dict[self.answer_key_name])
        answer_dict['PAGE'] = list(set(page_source))
        answer_dict['QUESTION'] = question
        answer_dict['ANSWER_LENGTH'] = answer_length
        answer_dict['USED_CHUNKS'] = used_chunks
        self.user_answers.append(answer_dict)

        return answer_dict, docs
