import langchain
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA, SimpleSequentialChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
from langchain.prompts import PromptTemplate
from config import LLM, CHROMA_CLIENT, CHAT_LLM, EMBEDDING_FUNC, VERBOSE, llm_chain
from CaseInfo import CaseInfo

from langchain.retrievers.multi_query import MultiQueryRetriever

def docstoreReactAgent(collection_name:str, query:str)->str:
    # db = Chroma(client=CHROMA_WEB_CLIENT, collection_name="law-docs")
    # docstore = DocstoreExplorer(db)
    print(collection_name, query)
    docstore = DocstoreExplorer(CaseInfo(collection_name))     # only one that work is Wikipedia, which is a wrapper class.
    tools = [
        Tool(
            name="Search",
            func=docstore.search,
            description="useful when never you need to find information about anything",
            # given search term, find a document
        ),
        Tool(
            name="Lookup",
            func=docstore.lookup,
            description="useful for when you need to ask with lookup",
            # find data within a document
        )
    ]
    react = initialize_agent(tools, CHAT_LLM, agent=AgentType.REACT_DOCSTORE, verbose=True)
    # query = "原告方基本信息"
    langchain.debug = True
    res= react.run(query)
    langchain.debug = False
    print(res)
    return res

# print(docstoreReactAgent(db="", query="案件名称？"))
def retrievalQAChain(collection_name:str, query:str):
    # format question
    db = Chroma(client=CHROMA_CLIENT, collection_name=collection_name, embedding_function=EMBEDDING_FUNC)
    prompt_temp = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say nothing and leave the answer blank. 

    {context}

    Question: {question}
    Examples:
    SYSTEM: the answers are plaintiff is Cisco Co., and defendant is Goo Ltd.
    OUTPUT: {{plaintiff: Cisco Co., address: 123 Main Street, CEO: John Smith}}
            {{defendant:Goo Ltd., phone: 312-2334-576, CEO: Charlie Brown}}

    Answer all questions in Chinese and export result in JSON format as the examples, use keyword in the question
    as KEY and result as VALUE in the JSON output"""

    PROMPT = PromptTemplate(template=prompt_temp, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(
        CHAT_LLM, 
        chain_type="stuff",
        retriever=db.as_retriever(),
        # return_source_documents=True,
        chain_type_kwargs = {"prompt": PROMPT},
        #  chain_type_kwargs={
        #     "question_prompt": question_prompt,
        #     "refine_prompt" : refine_prompt
        # },
        )
    # chain_type_kwargs only acceptable 'prompt' for STUFF chain. 
    # Refine chain accept more prompts: question_prompt, refine_prompt

    res = qa({"query": llm_chain("refine the following question in Chinese," + query)})
    print(res)
    return res

# res = retrievalQAChain("huggingface", "根据所提供资料，分别确定原告方及被告的基本信息。如当事人是公民（自然人），应写明姓名、性别、民族、出生年月日、住址、身份证号码、联系方式；当事人如是机关、团体、企事业单位，则写明名称、地址、统一社会信用代码、法定代表人姓名、职务")
# res = retrievalQAChain("5ACIVM0ewbQdqpgVtXhO3PW9QsJ", "refine my question below. \n\n find full name of the defendant")
# res = retrievalQAChain("5ACIVM0ewbQdqpgVtXhO3PW9QsJ", "Tell me what the plaintiff is suing for.")

def getTaskList(query:str):
    return llm_chain("Seperate the following text into list of wrong doings by the defendant. Export the content in an array. Quote the original text directly." + query)
    # return llm_chain("把下文中杭州栖溪对杭州阿家造成损失的事实逐条列印出来，使用原文内容即可。 " + query)

res = '答：根据原告杭州阿家的陈述，被告杭州栖溪对其造成的经济损失主要有以下几个方面：\n\n1. 杭州栖溪在双方签署《店铺租赁合同》之后，未取得规划等手续违法搭建了巨大的天桥，影响了阿家公司的正常经营，导致阿家无法正常使用租赁房屋，进而导致解除合同。\n\n2. 杭州栖溪在经营上未兑现招商承诺，与其在《西溪天堂商业街深化方案》和《项目介绍》中展示的设计理念和风格差别巨大，导致阿家公司为迎合商业街整体风格投入的资金和精力不仅无任何回报，还损失惨重。\n\n3. 杭州栖溪在管理上存在许多漏洞，造成整个景区经营不善，给阿家公司造成了巨额经济损失。\n\n4. 杭州栖溪以自身利益为考虑，收取高额租金，且未对其本职工作予以改进，甚至擅自对商业街进行违法建筑，使阿家公司的经营雪上加霜。\n\n根据以上事实，阿家公司可能会提出以下诉讼请求：\n\n1. 请求杭州栖溪赔偿因违法搭建天桥而导致的经营损失。\n\n2. 请求杭州栖溪赔偿因未兑现招商承诺而导致的投资损失。\n\n3. 请求杭州栖溪赔偿因管理不善而导致的经营损失。\n\n4. 请求杭州栖溪赔偿因擅自进行违法建筑而导致的经营损失。'

# subtasks = getSubTask(res)

def getLawDocs(query:str):
    # check Law database to find supporting materials for the point
    
    db = Chroma(client=CHROMA_CLIENT, collection_name='huggingface', embedding_function=EMBEDDING_FUNC)
    prompt_temp = """ use the following pieces of context. If nothing is found, just leave the answer blank. 

    {context}

    Argument: {question}
    Give your answer in the same language as the argument."""

    # Export result in JSON format. Using "FACT" as key to indicate fact and "REQUEST" as key to indicate the corresponding compensation request.
    # Example:
    # {{fact: the defendant costed $100,000 lost to the plaintiff, request: plaintiff requires $120,000 as compensation. }}
    # {{fact: the defendant close road to access the facility, request: plaintiff requires $20,000 compensation due to incapcity of normal operation. }}

    PROMPT = PromptTemplate(template=prompt_temp, input_variables=["context", "question"])
    CHAT_LLM.temperature = 0
    qa = RetrievalQA.from_chain_type(
        CHAT_LLM, 
        chain_type="stuff",
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs = {"prompt": PROMPT},
    )
    res = qa({"query": query})
    CHAT_LLM.temperature = 0
    print(res)

res = '杭州栖溪在双方签署《店铺租赁合同》之后，未取得规划等手续违法搭建了巨大的天桥，影响了阿家公司的正常经营，导致阿家无法正常使用租赁房屋，进而导致解除合同。'
# print(llm_chain("下述问题会涉及到哪几部相关法律？" +res))
# print(llm_chain("从中华人民共和国合同法，中华人民共和国房地产管理法，中华人民共和国侵权责任法。引用与下述问题相关的条例以支持原告阿家的诉求。" +res))
# getLawDocs("查询中华人民共和国国民经济和社会发展第十个五年相关的内容。")
