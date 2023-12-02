import textwrap
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


 

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

class query_youtube:
    def __init__(self, video_url):
        self.video_url = video_url

    def create_db_from_youtube_video_url(self):
        loader = YoutubeLoader.from_youtube_url(self.video_url)
        transcript = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript)

        db = FAISS.from_documents(docs, embeddings)
        self.db = db
    
    def get_response_from_query(self, query, k=4):
        """
        gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
        the number of tokens to analyze.
        """
    
        docs = self.db.similarity_search(query, k=k)
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
        self.response = response
        self.docs = docs

    def print_docs(self):
        for doc in self.docs:
            print(doc.page_content)
    
    def print_response(self):
        print(textwrap.fill(self.response, 80))





