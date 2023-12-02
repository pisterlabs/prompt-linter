import os

from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
# from langchain.chains import APIChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFPlumberLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from .functions.block_card import BlockCardTool
from .functions.open_account import OpenAccountTool


class RaiPayChatbot():

    def __init__(self, max_tokens, temperature, verbose):
        # Load environment variables from the .env file
        load_dotenv()

        self.pdf_path = os.getenv("PDF_PATH")
        self.conversation_chat = self.load_assets(max_tokens, temperature, verbose)

    def query(self, question):
        answer = self.conversation_chat({'input': question})
        return answer['output']

    def load_assets(self, max_tokens, temperature, verbose):
        # extra text from pdfs and generate text chunks
        chunks = self.extract_text_from_pdf()

        # generate embedding database
        vector_database = self.generate_vector_database(chunks)

        # create conversation chain
        conversation_chat = self.get_conversation_chain(vector_database, max_tokens, temperature, verbose)

        return conversation_chat

    def extract_text_from_pdf(self):
        # Load pdf text
        loader = PDFPlumberLoader(self.pdf_path)

        # Split the pdf into chunks to save space in the context window and generate a vector database
        chunk_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = loader.load_and_split(text_splitter=chunk_splitter)

        return chunks

    def generate_vector_database(self, text_chunks):
        # Generating a local database embedding databased based on the text chunks
        vector_database = FAISS.from_documents(
            documents=text_chunks,
            embedding=OpenAIEmbeddings()
        )
        return vector_database

    def get_conversation_chain(self, vector_database, max_tokens=None, temperature=0, verbose=False):
        tools = [
            create_retriever_tool(
                vector_database.as_retriever(),
                "raipay_app_faq_database",
                description="Searches and returns documents regarding RaiPay app. Input: send the user input."
            ),
            BlockCardTool(),
            OpenAccountTool()
        ]

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-0613",
            max_tokens=max_tokens,
            temperature=temperature,  # As we want the chat to answer specific question from the text, it's better to control the level of randomness
        )

        agent_executor = create_conversational_retrieval_agent(
            llm=llm,
            tools=tools,
            verbose=verbose,
            remember_intermediate_steps=True,
            max_token_limit=2000
        )

        ## Old chatbot for the 1st task that was updated to be an agent and understand mutiple functions

        # Set LLM and memory buffer
        # conversation_chain = ConversationalRetrievalChain.from_llm(
        #     llm=ChatOpenAI(
        #         model_name="gpt-3.5-turbo-0613",
        #         max_tokens=None,  # we don't want to trunking the text
        #         temperature=0,  # As we want the chat to answer specific question from the text, it's better to control the level of randomness
        #     ),
        #     retriever=vector_database.as_retriever(),
        #     memory=memory,
        #     functions=functions_defs
        # )

        return agent_executor
