from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.output_parsers import RegexParser
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

def process_docs():
    loader = DirectoryLoader(f'docs', glob="./*", show_progress=True, use_multithreading=True)
    documents = loader.load()
    chunk_size_value = 1000
    chunk_overlap=100
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_value, chunk_overlap=chunk_overlap, length_function=len)
    texts = text_splitter.split_documents(documents)

    docembeddings = FAISS.from_documents(texts, OpenAIEmbeddings())
    docembeddings.save_local("llm_faiss_index")
    docembeddings = FAISS.load_local("llm_faiss_index",OpenAIEmbeddings())
    
    return docembeddings

def setup_qa_chain():
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    This should be in the following format:

    Question: [question here]
    Helpful Answer: [answer here]
    Score: [score between 0 and 100]

    Begin!

    Context:
    ---------
    {context}
    ---------
    Question: {question}
    Helpful Answer:"""
    output_parser = RegexParser(
        regex=r"(?s)(.*?)\s*Score:\s*(.*)",
        output_keys=["answer", "score"],
    )
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        output_parser=output_parser
    )
    # portkey_headers = {
    # "x-portkey-api-key": "portkey-api-key",
    # "x-portkey-mode": "proxy openai"
    # }
    llm = OpenAI(temperature=0.4, max_tokens=1000)
    chain = load_qa_chain(llm=llm, chain_type="map_rerank", return_intermediate_steps=True, prompt=PROMPT)
    return chain

def getanswer(docembeddings, chain, query):
    relevant_chunks = docembeddings.similarity_search_with_score(query, k=4)
    chunk_docs=[]
    
    for chunk in relevant_chunks:
        chunk_docs.append(chunk[0])
    results = chain({"input_documents": chunk_docs, "question": query})
    text_reference=""
    
    for i in range(len(results["input_documents"])):
        text_reference += results["input_documents"][i].page_content
    text_reference = text_reference.replace("\n", "")
    output={"Answer":results["output_text"],"Reference":text_reference}
    output = {k: v.replace('\n', '') for k, v in output.items()}
    
    return output
