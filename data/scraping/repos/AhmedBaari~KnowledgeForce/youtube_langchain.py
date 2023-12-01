# %%
# Import statements
import os 
from dotenv import load_dotenv, find_dotenv

from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # cuz thousands of lines
from langchain.chains import LLMChain
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.agents import load_tools, initialize_agent, AgentType

# %% [markdown]
# 

# %%

# Load environment variables
load_dotenv(find_dotenv())

# API Key
google_api_key = os.environ.get('GOOGLE_API_KEY') 

# %% [markdown]
# Set up LLM

# %%
# Set up LLM
llm = GooglePalm(google_api_key=google_api_key)
llm.temperature = 0.7

embeddings = GooglePalmEmbeddings()

# %%
def create_vector_db_from_youtube_url(video_url) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50) #overlap: last 100 words of doc 1 is first 100 words in doc 2

    docs = text_splitter.split_documents(transcript)

    
    db = FAISS.from_documents(docs, embeddings)

    ## We cannot send so much information to openai, so we will do a similarity search
    ## and only send the most similar documents

    

    return db

# %%

# %%
def get_response_from_query(db, query, k=4):
    """
    PaLM can handle up to 4096 tokens. 
    Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    print(docs_page_content[0])

    # Set up LLM
    llm = GooglePalm(google_api_key=google_api_key)
    llm.temperature = 0.5
    print("LLM is ", llm)

    # Set up prompt template
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )


    print("Prompt is ", prompt)

    # Set up chain
    chain = LLMChain(llm=llm, prompt=prompt)

    print("Chain is: ",chain)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    (print(response))
    return response, docs


# %%



