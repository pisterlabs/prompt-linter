#Running this file will cost about $0.07
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

python_environment = find_dotenv()
load_dotenv(python_environment)

netanyahu_interview_url = "https://www.youtube.com/watch?v=XpC7SVDXimg"
embeddings = OpenAIEmbeddings()

# Cost: ~2c
def create_database_from_youtube_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(transcript)
    database = FAISS.from_documents(split_docs, embeddings)
    return database

#Cost: ~5c
def get_response_from_query(database, query, vector_count=8):
    matching_docs = database.similarity_search(query, k=vector_count)
    doc_content = "\n---\n".join([doc.page_content for doc in matching_docs])
    
    gpt = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
    system_template = """
        You are an intelligent chatbot assistant designed to answer questions about youtube videos 
        based on excerpts of the video's transcript within triple backticks below.
        ```
        {doc_content}
        ```

        Instructions:
        - Only use the factual information from the transcript to answer the question.
        - If you're unsure of an answer, you can say "I don't know" or "I'm not sure"
        """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)    
    
    human_template = "Answer the following question that has been placed within angle brackets <{query}>"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    llmChain = LLMChain(llm=gpt, prompt=chat_prompt)
    response = llmChain.run(query=query, doc_content=doc_content)
    return response
    
    
    
database = create_database_from_youtube_url(netanyahu_interview_url)
response = get_response_from_query(database, "How does Benjamin think peace in the middle east can be achieved?")
print(response)