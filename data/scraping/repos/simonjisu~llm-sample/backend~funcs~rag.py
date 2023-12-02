import os
import pandas as pd
from loguru import logger
from pathlib import Path
from funcs import summarize as sm

from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# output parser
from pydantic import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser

class ClassificationAnswer(BaseModel):
    answer: int = Field(description="The answer to the classification question.")
        
CLS_PARSER = PydanticOutputParser(pydantic_object=ClassificationAnswer)

def question_answering(ticker: str, query: str) -> str:
    # load df_ticker cluster and summarized text
    df_ticker_path = Path(f"data/{ticker}_clustered.csv").resolve()
    if not df_ticker_path.exists():
        logger.info(f"[question_answering] {df_ticker_path} does not exist, start to create the summary file")
        sm.summarize_ticker(ticker)
    df_ticker = pd.read_csv(df_ticker_path)
    n_cluster = df_ticker['cluster'].nunique()
    choices = str(list(range(n_cluster)))

    with open(f"data/{ticker}_summarized.txt", 'r') as f:
        summarized_text = f.read()
    
    # find the where the query belongs to from the summary document
    template = """Here is a summary of {ticker}'s financial report and useful information:
    {summarized_text}
    Based on this summarized_text, please classify where the question is belong to refer to the user's question: {question}
    Answer with only one of the cluster numbers({choices})\n{format_instructions}
    Answer:"""
    prompt = PromptTemplate.from_template(
        template, 
        partial_variables={"format_instructions": CLS_PARSER.get_format_instructions()}
    )
    llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_KEY"], model="gpt-4")
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=CLS_PARSER)
    results = chain.run(ticker=ticker, summarized_text=summarized_text, question=query, choices=choices)
    logger.info(f"[question_answering] finished classifying the query belongs to which cluster: {results}")

    # get the document from the cluster
    cluster = results.answer
    matched_index = df_ticker['cluster']==cluster
    docs = [] 
    for i, row in df_ticker.loc[matched_index, ['message', 'page']].iterrows():
        docs.append(
            Document(page_content=row['message'], metadata={'page': row['page']})
        )
    logger.info(f"[question_answering] finished loading filtered docs")

    # search similar docs based on user's query
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"])
    faiss_index = FAISS.from_documents(docs, embedding=embeddings)
    docs = faiss_index.similarity_search(query, k=5)
    logger.info(f"[question_answering] finished searching similar docs")

    # QA
    template = """You are an CFO in {ticker}. Here are a set of documents:
    {docs}
    Based on this list of docs, please answer the user's question: {question}
    Answer:"""
    prompt = PromptTemplate.from_template(template)
    chain = LLMChain(llm=llm, prompt=prompt)
    qa_answer = chain.run(ticker=ticker, docs=docs, question=query)
    logger.info(f"[question_answering] finished QA")

    return qa_answer