from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader, S3DirectoryLoader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores.faiss import FAISS
import pickle
import os
import requests
from dotenv import load_dotenv

openai_api_key = os.getenv('OPENAI_API_KEY')



class QnABot:
    def __init__(
        self,
        directory: str,
        index: str | None = None,
        model: str | None = None,
        temperature=0,
    ):
        # Initialize the QnABot by selecting a model, creating a loader, and loading or creating an index
        self.select_model(model, temperature)
        self.create_loader(directory)
        self.load_or_create_index(index)

        # Load the question-answering chain for the selected model
        self.chain = load_qa_with_sources_chain(self.llm)

    def select_model(self, model: str | None, temperature: float):
        # Select and set the appropriate model based on the provided input
        if model is None or model == "gpt-4":         
            self.llm = ChatOpenAI(temperature=temperature)


    def create_loader(self, directory: str):
        # Create a loader based on the provided directory (either local or S3)
        if directory.startswith("s3://"):
            self.loader = S3DirectoryLoader(directory)
        else:
            self.loader = DirectoryLoader(directory, recursive=True)

    def load_or_create_index(self, index_path: str | None):
        # Load an existing index from disk or create a new one if not available
        if index_path is not None and os.path.exists(index_path):
            # print("Loading path from disk...")
            with open(index_path, "rb") as f:
                self.search_index = pickle.load(f)
        else:
            print("Creating index...")
            # model_name="multi-qa-mpnet-base-cos-v1" , paraphrase-multilingual-mpnet-base-v2, all-mpnet-base-v2
            embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
            # print(embeddings)
            # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            self.search_index = FAISS.from_documents(
                self.loader.load_and_split(), embeddings 
            )

    def save_index(self, index_path: str):
        # Save the index to the specified path
        print("Saving pickle")
        with open(index_path, "wb") as f:
            pickle.dump(self.search_index, f)

    def truncate_to_word_limit(self, text, word_limit):
        words = text.split()
        if len(words) > word_limit:
            text = ' '.join(words[:word_limit])
        return text

    def get_answer(self, question, k=4):
        input_documents = self.search_index.similarity_search(question, k=k)
        input_documents_str = self.truncate_to_word_limit('\n'.join([doc.page_content for doc in input_documents]), 600)
        API_URL = "https://flow.nuiq.ai/api/v1/prediction/ff25814e-5bb8-4192-8266-6e740be9233d"
        headers = {"Authorization": "Bearer Aj05iilwEqjLQRnlHMtWerW1VqANw3ugzls8XXK0t0o="}
        sysprompt = "<s>[INST] <<SYS>>\nYou are Q Notes. You take therapy session transcripts and return well formatted and detailed notes as requested. Only return the note and nothing else.  \n Use bulleted lists. \n Here is the transcript from which you should answer these questions: \n```" + input_documents_str + "\n```<</SYS>>"
        num_words = len(sysprompt.split())

        print(f"Number of words in system prompt: {num_words}")

        payload = {
            "question": question + "[/INST]",
            "overrideConfig": {
                "systemMessagePrompt": sysprompt,
                # "maxTokens": 1000,
                
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        print(response.json())
        return response.json()