"""
Initialize QA bot.
"""
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

def init_bot(vectordb, OPENAI_API_KEY):
    """Initialize the doc-talk bot."""

    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    bot = RetrievalQA.from_chain_type(llm=llm,
                                      chain_type="stuff",
                                      retriever=vectordb.as_retriever())
    
    return bot