import json

from kendra_index_retriever import KendraIndexRetriever
from langchain.prompts import PromptTemplate
from langchain.agents import BaseSingleActionAgent, AgentOutputParser, LLMSingleActionAgent, AgentExecutor
from langchain.chains import RetrievalQA


class KendraLLMRetrieverQAChain():

    def __init__(self, KendraIndexId,
                 KendraLanguageCode,
                 AWSRegion,
                 Llm):

        self.KENDRA_INDEX_ID = KendraIndexId
        self.REGION = AWSRegion
        self.KENDRA_LANG_CODE = KendraLanguageCode


        TEMPLATE_1 = '''
        Instruction:现在你是一个只能依据给定信息回答问题的智能助手，你可以基于下面提供的背景知识，有逻辑地对相关问题进行解答。但如果你发现提供的背景知识没有问题相关的信息，请回答\"不知道\"。
        背景知识:{context}。
        问题:{question}。
        解答:
        '''
        
        PROMPT_1 = PromptTemplate(
            template=TEMPLATE_1, input_variables=["context", "question"]
        )

        self.kendra_retriever = KendraIndexRetriever(kendraindex=self.KENDRA_INDEX_ID,
                awsregion=self.REGION,
                return_source_documents=True,
                langcode=self.KENDRA_LANG_CODE
            )

        self.sm_llm=Llm

        chain_type_kwargs = {"prompt": PROMPT_1}

        self.kendra_chain_qa = RetrievalQA.from_chain_type(
            self.sm_llm,
            chain_type="stuff",
            retriever=self.kendra_retriever,
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
            )
    
    
    def get_chain(self):
        return self.kendra_chain_qa
    
    def get_raw_retriever(self):
        return self.kendra_retriever
    
    def get_chain_QA(self, query=''):
        resp = self.kendra_chain_qa(query)
        resp_as_pt = f'''问题:{resp['query']}。解答:{resp['result']}'''
        return resp_as_pt
        