from dotenv import load_dotenv 

import VectorStore
import LLM
import Chain
import Aggregator
import os
import streamlit as st
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


from langchain import PromptTemplate


from langchain.chains import LLMChain

def main():
    load_dotenv()
    folder_location = os.getcwd() + "\Hackathon\Hackathon\\resources"

    # Define UI elements
    st.set_page_config(page_title="Stock Analysis", page_icon="📈")
    st.header("Stock Analysis")
    stock = st.sidebar.selectbox("Select Stock", ("Microsoft", "Google", "NVIDIA", "Tesla", "Amazon"))

    # Storing data in the store
    embeddings = VectorStore.HuggingFaceEmbeddings()
    VectorStore.create_dataset(folder_location, embeddings)

    if stock:
        print("Starting processing")
        aggcontext = GenerateAggregatedContext("What are positive, negative and neutral news about " + stock + " ?", embeddings)
        print("Aggregated Context:")
        score = GenerateSentimentScore(stock, aggcontext)
        st.header("Aggregated Context")
        st.write(aggcontext)
        st.header("Sentiment Score")
        st.write(score)
        #print(aggcontext)

def GenerateAggregatedContext(query, embeddings):
    db = VectorStore.get_dataset(embeddings)
    retriever = VectorStore.getDataRetriever(db)
    llm = LLM.createLLM("gpt-35-turbo-16k")
    qa = Chain.createRetrivalChain(llm, retriever)
    AggregatedContext = Aggregator.getAggregatedContext(qa, query)
    return AggregatedContext

def GenerateSentimentScore(stockname, context):
    #prompt_template = PromptTemplate.from_template("")    
    prompt = PromptTemplate(
        input_variables=["StockName", "content"],
        template="You are a expert financial analyst. The stockscore is between 0 and 100. Here, more positive the news, higher the stockscore. Provide a stockscore of {StockName} stock based on news = {content}. Only give stockscore in the output.",
    )  
    llm = LLM.createLLM("gpt-35-turbo-16k")
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(StockName=stockname, content = context)
    return response


if __name__ == "__main__":

    main()

