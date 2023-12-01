import threading
import dotenv
from tkinter import messagebox
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

class YouTubeProcessor:

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""
        self.map_prompt = PromptTemplate.from_template(self.map_template)
        self.map_chain = LLMChain(llm=self.llm, prompt=self.map_prompt)
        self.reduce_template = """The following is set of summaries:
{doc_summaries}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""
        self.reduce_prompt = PromptTemplate.from_template(self.reduce_template)
        self.reduce_chain = LLMChain(llm=self.llm, prompt=self.reduce_prompt)
        self.combine_documents_chain = StuffDocumentsChain(
            llm_chain=self.reduce_chain, document_variable_name="doc_summaries"
        )
        self.reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=self.combine_documents_chain,
            collapse_documents_chain=self.combine_documents_chain,
            token_max=4000,
        )
        self.map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=self.map_chain,
            reduce_documents_chain=self.reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

    def process_youtube_url(self, url, chunk_size, callback):
        if "youtu" not in url:
            messagebox.showerror("Error", "Invalid YouTube URL!")
            return
        thread = threading.Thread(target=self.run_processing, args=(url, chunk_size, callback))
        thread.start()

    def run_processing(self, url, chunk_size, callback):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        docs = loader.load()
        split_docs = text_splitter.split_documents(docs)
        result = self.map_reduce_chain.run(split_docs)
        callback(result)
