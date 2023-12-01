import os
import textwrap
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import scraper

# Load environment variables
load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

# Variable Declaration
# api_key = os.environ["OPENAI_API_KEY"]
outputDirectory = "output/"
chunkSize = 1000
chunkOverlap = 10
kSize = 4

def prepareUrls():
    videoUrls = []
    file = open(outputDirectory + "urls.txt", "r")
    for line in file:
        videoUrls.append(line)
    return videoUrls	

def createDB(videoUrls, chunkSize=chunkSize, chunkOverlap=chunkOverlap):
    docsContainer = []
    for url in videoUrls:
        loader = YoutubeLoader.from_youtube_url(url)
        transcript = loader.load()
        textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = (textSplitter.split_documents(transcript))
        
        
    # for docs in docsContainer:
    #     print(docs)

    # db = FAISS.from_documents(docsContainer, embeddings)
    
    # return db


def getResponse(db, query, k=kSize):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs

def exportResponse(response):
    outputFile = open(outputDirectory + "output.txt", "w")
    for line in response:
        outputFile.write(line)
    
    outputFile.close()
    outputFile.truncate(0)


# youtubeChannel = input("Enter the Youtube Channel URL: ")
# urlList = scraper.getUrls(youtubeChannel)
videoUrls = prepareUrls()
db = createDB(videoUrls)
query = input("Enter your question: ")
response, docs = getResponse(db, query)

print(textwrap.fill(response, width=50))
exportResponse(response)