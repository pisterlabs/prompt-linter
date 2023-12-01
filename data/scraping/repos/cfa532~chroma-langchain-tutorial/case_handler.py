from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

# from langchain.callbacks import get_openai_callback
from config import CHROMA_CLIENT, EMBEDDING_FUNC, CHAT_LLM, llm_chain, LegalCase, LAW_COLLECTION_NAME

def analyse_wrongdoing(my_case:LegalCase, query:str):
    law_db = Chroma(client=CHROMA_CLIENT, collection_name=LAW_COLLECTION_NAME, embedding_function=EMBEDDING_FUNC)
    # figure out the laws violated
    laws = llm_chain("下述问题会涉及到哪几部相关法律？" +query)
    print("Laws: " + laws)
    for l in laws:
        res=query_docstore(law_db, query+" 触及 "+l+" 的那些具体条款？在回答中引用具体条款内容。")
        print(res)

# Give text to create a in memory vector DB and answer query based on its content
def init_case(text):
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=['.', '\n\n', '\n', ',', '。','，'])
    chunks.extend(text_splitter.split_text(text))

    # Have to appoint to the remote server if Chroma is running on different machine from Flask
    file_db = Chroma.from_texts(chunks, embedding=EMBEDDING_FUNC, client=CHROMA_CLIENT)
    res = {}
    o = query_docstore(file_db, "找出原告名称。")
    res.update({"plaintiff": o["result"]})
    o = query_docstore(file_db, "找出被告名称。")
    res.update({"defendant": o["result"]})

    # rate limit maybe triggered.
    o = query_docstore(file_db, "给这个案件起一个标题。")
    res.update({"title": o["result"]})
    
    return res

# Try to figure out the purpose of the lawsuit.
def get_request(db_retriever, query:str, temperature=0.0):
    prompt_temp = """ Use the following pieces of context to answer question at the end. If you don't know the answer, just say nothing and leave the answer blank. Try to find more than a couple of faults of the defendant.

    {context}

    Question: {question}
    Answer all questions in Chinese."""

    # Export result in JSON format. Using "FACT" as key to indicate fact and "REQUEST" as key to indicate the corresponding compensation request.
    # Example:
    # {{fact: the defendant costed $100,000 lost to the plaintiff, request: plaintiff requires $120,000 as compensation. }}
    # {{fact: the defendant close road to access the facility, request: plaintiff requires $20,000 compensation due to incapcity of normal operation. }}

    PROMPT = PromptTemplate(template=prompt_temp, input_variables=["context", "question"])
    CHAT_LLM.temperature = temperature
    qa = RetrievalQA.from_chain_type(
        CHAT_LLM, 
        chain_type="stuff",
        retriever=db_retriever,
        # return_source_documents=True,
        chain_type_kwargs = {"prompt": PROMPT},
    )
    # refine the query to get a better solution
    query = llm_chain("refine the following question in Chinese," + query)
    res = qa({"query": query})
    CHAT_LLM.temperature = 0
    return res, query

# getRequest("huggingface", "列举杭州栖溪对杭州阿家造成的经济损失事实，并且提出合理的诉讼请求。")

def get_argument(collection_name:str, query:str, temperature=0.0):
    db = Chroma(client=CHROMA_CLIENT, collection_name=collection_name, embedding_function=EMBEDDING_FUNC)
    prompt_temp = """ Use the following pieces of context to answer question at the end. If you don't know the answer, just say nothing and leave the answer blank. Try to find more than a couple of faults of the defendant.

    {context}

    Question: {question}
    Answer all questions in Chinese."""

    PROMPT = PromptTemplate(template=prompt_temp, input_variables=["context", "question"])
    CHAT_LLM.temperature = temperature
    qa = RetrievalQA.from_chain_type(
        CHAT_LLM, 
        chain_type="stuff",
        retriever=db.as_retriever(),
        # return_source_documents=True,
        chain_type_kwargs = {"prompt": PROMPT},
    )
    query = llm_chain("refine the following question in Chinese," + query)
    res = qa({"query": query})
    CHAT_LLM.temperature = 0
    return res, query

def query_docstore(db_retriever, query:str):
    prompt_temp = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say nothing and leave the answer blank. 

    {context}

    Question: {question}

    Export results formatted like the above examples, in key:value pairs, where both the key and value shall be in Chinese. If there is not enough information to answer the query, still export a key:value pair, but leave an empty space as value.
    
    Give all replies in Chinese."""

    PROMPT = PromptTemplate(template=prompt_temp, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(
        CHAT_LLM, 
        chain_type="stuff",
        retriever=db_retriever,
        return_source_documents=True,
        chain_type_kwargs = {"prompt": PROMPT},
    )
    refined_query = llm_chain("refine the following question in Chinese," + query)
    res = qa({"query": refined_query})
    print("get_Json: ", res)
    # the first returned value is refined question, the 2nd is the result.
    return {"query":res["query"], "result":res["result"]}

def get_basic_info(db_retriever, query:str):
    prompt_temp = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say nothing and leave the answer blank. 

    {context}

    Question: {question}

    Examples:
    SYSTEM: the answers are plaintiff is Cisco Co., and defendant is Goo Ltd.
    OUTPUT: plaintiff: Cisco Co., 
            address: 123 Main Street, 
            CEO: John Smith,
            defendant: Goo Ltd., 
            phone number: 312-2334-576,


    Export results formatted like the above examples, in key:value pairs, where both the key and value shall be in Chinese. If there is not enough information to answer the query, still export a key:value pair, but leave an empty space as value.
    
    Give all replies in Chinese."""

    PROMPT = PromptTemplate(template=prompt_temp, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(
        CHAT_LLM, 
        chain_type="stuff",
        retriever=db_retriever,
        # return_source_documents=True,
        chain_type_kwargs = {"prompt": PROMPT},
    )
    refined_query = llm_chain("refine the following question in Chinese," + query)
    res = qa({"query": refined_query})
    # print("get_Json: ", res)
    # the first returned value is refined question, the 2nd is the result.
    return {"query":res["query"], "result":res["result"]}