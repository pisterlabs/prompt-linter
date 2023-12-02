import os
import tkinter as tk
from tkinter import filedialog

import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
)

from .gui import Tooltip, Settings, Conversation
from .utils import tokens2price, text2tokens

DOCUMENT_LOADERS = {
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".pdf": (PDFMinerLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".csv": (CSVLoader, {"encoding": "utf8"}),
}

class DocGPT:
    def __init__(
        self,
        kanu,
        openai_key,
        model,
        temperature,
        prompt,
        default_chunk_size,
        default_chunk_overlap,
        new_database_directory="",
        document_directory="",
        existing_database_directory="",
    ):
        self.kanu = kanu
        self.model = model
        self.temperature = temperature
        self.prompt = prompt
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        os.environ["OPENAI_API_KEY"] = openai_key
        self.settings = Settings(self)
        self.conversation = Conversation(self)
        self.tokens = 0
        self.price = 0
        self.new_database_directory = new_database_directory
        self.document_directory = document_directory
        self.existing_database_directory = existing_database_directory

    def run(self):
        self.kanu.container.pack_forget()
        self.kanu.container = tk.Frame(self.kanu.root)
        self.kanu.container.pack()
        self.kanu.container.bind_all("<Return>", lambda event: self.send_message())
        self.kanu.container.focus_set()
        l = tk.Label(self.kanu.container, text="DocGPT")
        l.grid(row=0, column=0, columnspan=3)
        b = tk.Button(self.kanu.container, text="Go back", command=lambda: self.kanu.config_docgpt())
        b.grid(row=1, column=0)
        b = tk.Button(self.kanu.container, text="Reload", command=lambda: self.run())
        b.grid(row=1, column=2)
        m = tk.Message(self.kanu.container, width=300, text="Option 1. Create a new database")
        m.grid(row=2, column=0, columnspan=3)
        l = tk.Label(self.kanu.container, text="Document ⓘ:")
        Tooltip(l, "Directory containing documents for the database.")
        l.grid(row=3, column=0) 
        self.document_label = tk.Label(self.kanu.container, text="Not selected", fg="red")
        self.document_label.grid(row=3, column=1)
        b = tk.Button(self.kanu.container, text="Browse", command=self.specify_document_directory)
        b.grid(row=3, column=2)
        l = tk.Label(self.kanu.container, text="Database ⓘ:")
        Tooltip(l, "Directory where the database will be stored.")
        l.grid(row=4, column=0)       
        self.new_database_label = tk.Label(self.kanu.container, text="Not selected", fg="red")
        self.new_database_label.grid(row=4, column=1)
        b = tk.Button(self.kanu.container, text="Browse", command=self.specify_new_database_directory)
        b.grid(row=4, column=2)
        l = tk.Label(self.kanu.container, text="Chunk size ⓘ:")
        Tooltip(l, "The maximum number of characters in each chunk.")
        l.grid(row=5, column=0)
        self.chunk_size = tk.IntVar(self.kanu.container, value=self.default_chunk_size)
        e = tk.Entry(self.kanu.container, textvariable=self.chunk_size)
        e.grid(row=5, column=1, columnspan=2)
        l = tk.Label(self.kanu.container, text="Chunk overlap ⓘ:")
        Tooltip(l, "The number of overlapping characters between adjacent chunks.")
        l.grid(row=6, column=0)
        self.chunk_overlap = tk.IntVar(self.kanu.container, value=self.default_chunk_overlap)
        e = tk.Entry(self.kanu.container, textvariable=self.chunk_overlap)
        e.grid(row=6, column=1, columnspan=2)
        self.option1_button = tk.Button(self.kanu.container, text="Go with Option 1", command=self.go_with_option1)
        self.option1_button.grid(row=7, column=0, columnspan=3)
        self.option1_button["state"] = tk.DISABLED
        m = tk.Message(self.kanu.container, width=300, text="Option 2. Use an existing database")
        m.grid(row=8, column=0, columnspan=3)
        l = tk.Label(self.kanu.container, text="Database ⓘ:")
        Tooltip(l, "Directory where the database is stored.")
        l.grid(row=9, column=0)
        self.existing_database_label = tk.Label(self.kanu.container, text="Not selected", fg="red")
        self.existing_database_label.grid(row=9, column=1)
        b = tk.Button(self.kanu.container, text="Browse", command=self.specify_existing_database_directory)
        b.grid(row=9, column=2)
        self.option2_button = tk.Button(self.kanu.container, text="Go with Option 2", command=self.go_with_option2)
        self.option2_button.grid(row=10, column=0, columnspan=3)
        self.option2_button["state"] = tk.DISABLED
        if self.new_database_directory:
            self.new_database_label.configure(text=os.path.basename(self.new_database_directory), fg="lime green")
        if self.document_directory:
            self.document_label.configure(text=os.path.basename(self.document_directory), fg="lime green")
        if self.new_database_label["text"] != "Not selected" and self.document_label["text"] != "Not selected":
            self.option1_button["state"] = tk.NORMAL
        if self.existing_database_directory:
            self.existing_database_label.configure(text=os.path.basename(self.existing_database_directory), fg="lime green")
            self.option2_button["state"] = tk.NORMAL

    def query(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.db = Chroma(persist_directory=self.database_directory, embedding_function=OpenAIEmbeddings())
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=self.model, temperature=self.temperature),
            retriever=self.db.as_retriever(),
            memory=self.memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": PromptTemplate(template=self.prompt, input_variables=["context", "question"])}
        )
        self.conversation.page()

    def send_message(self):
        try:
            with get_openai_callback() as cb:
                response = self.qa(self.user_input.get())
                self.calculate_usage(cb)
            self.session.insert(tk.END, "You: " + self.user_input.get() + "\n", "user")
            self.session.insert(tk.END, "Bot: " + response["answer"] + "\n", "bot")
            self.chatbox.delete(0, tk.END)
        except openai.error.InvalidRequestError as e:
            error = str(e)
            if "Please reduce the length of the messages." in error:
                self.system.insert(tk.END, f"System: {error} You can also create a new chat session.\n", "system")
            else:
                raise

    def calculate_usage(self, cb):
        prompt_price = tokens2price(self.model, "prompt", cb.prompt_tokens)
        completion_price = tokens2price(self.model, "completion", cb.completion_tokens)
        self.price += prompt_price + completion_price
        self.tokens += cb.total_tokens
        message = f"System: Used {cb.prompt_tokens:,} prompt + {cb.completion_tokens:,} completion = {cb.total_tokens:,} tokens (total: {self.tokens:,} or ${self.price:.6f})."
        self.system.insert(tk.END, f"{message}\n", "system")

    def go_with_option1(self):
        self.database_directory = self.new_database_directory
        self.tokens = self.price = 0
        documents = []
        for root, dirs, files in os.walk(self.document_directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file_path)[1]
                if file_ext not in DOCUMENT_LOADERS:
                    continue
                loader_class, loader_kwargs = DOCUMENT_LOADERS[file_ext]
                loader = loader_class(file_path, **loader_kwargs)
                document = loader.load()
                documents.extend(document)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size.get(), chunk_overlap=self.chunk_overlap.get())
        texts = text_splitter.split_documents(documents)
        for text in texts:
            self.tokens += 2 * text2tokens("text-embedding-ada-002", text.page_content)
        self.price = tokens2price("text-embedding-ada-002", "embedding", self.tokens)
        db = Chroma.from_documents(texts, OpenAIEmbeddings(model="text-embedding-ada-002"), persist_directory=self.database_directory)
        db.add_documents(texts)
        db.persist()
        db = None
        self.existing = False
        self.query()

    def go_with_option2(self):
        self.database_directory = self.existing_database_directory
        self.tokens = self.price = 0
        self.existing = True
        self.query()

    def specify_document_directory(self):
        directory_path = filedialog.askdirectory()
        if not directory_path:
            return
        self.document_directory = directory_path
        self.document_label.configure(text=os.path.basename(directory_path), fg="lime green")
        if self.new_database_label["text"] != "Not selected":
            self.option1_button["state"] = tk.NORMAL

    def specify_new_database_directory(self):
        directory_path = filedialog.askdirectory()
        if not directory_path:
            return
        self.new_database_directory = directory_path
        self.new_database_label.configure(text=os.path.basename(directory_path), fg="lime green")
        if self.document_label["text"] != "No file selected":
            self.option1_button["state"] = tk.NORMAL

    def specify_existing_database_directory(self):
        directory_path = filedialog.askdirectory()
        if not directory_path:
            return
        self.existing_database_directory = directory_path
        self.existing_database_label.configure(text=os.path.basename(directory_path), fg="lime green")
        self.option2_button["state"] = tk.NORMAL

    def clear_session(self):
        self.existing = True
        self.tokens = self.price = 0
        self.query()
