"""Used to load a personna."""
import os
from pathlib import Path
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from .prompts_models import EMBEDDING_MODEL, embedding_chain, answering_chain, strict_answering_chain, check_chain

DEFAULT_PERSONAS_FOLDER='./personas'
CHUNK_SIZE=1000


class Persona:
    """
    Stores all the information about a given persona.
    """

    def list(personas_folder=DEFAULT_PERSONAS_FOLDER):
        """
        returns a list of the names of all personas available
        """
        return [f.name for f in os.scandir(personas_folder) if f.is_dir()]

    def __init__(self, name, use_strict=False, personas_folder=DEFAULT_PERSONAS_FOLDER):
        """
        takes the name of a persona and loads it into memory
        generates it if it has not been generated before
        """
        # persona information
        self.name = name 
        self.personas_folder = Path(personas_folder)
        self.persona_path = self.personas_folder / name
        self.is_strict = use_strict  
        # documents database
        if self._database_exist():
            self.database = FAISS.load_local(folder_path=self.persona_path, embeddings=EMBEDDING_MODEL)
        else:
            self.database = self._generate_database()
    
    # ----- DATABASE -----

    def _database_exist(self):
        """
        returns True if the vector database has already been built
        """
        index_faiss = self.persona_path / 'index.faiss'
        index_pkl = self.persona_path / 'index.pkl'
        return index_faiss.is_file() and index_pkl.is_file()

    def _generate_database(self):
        """
        builds the vector database from scratch and saves it in a folder
        returns it
        WARNING: this operation can be slow and costly in OpenAI embeddings costs
        """
        print(f"Generating the vector database for '{self.name}'.")
        # loading the documents
        raw_documents = []
        nb_documents = []
        for folder_name in ['texts_by', 'texts_about']:
            documents_path = self.persona_path / folder_name
            loader = DirectoryLoader(documents_path)
            raw_documents_folder = loader.load()
            nb_documents.append(len(raw_documents_folder))
            # label documents as being written by the persona
            for document in raw_documents_folder:
                document.metadata['written_by_persona'] = (folder_name == 'texts_by')
            raw_documents.extend(raw_documents_folder)
        print(f"   Loaded {len(raw_documents)} documents ({nb_documents[0]} by, {nb_documents[1]} about).")
        # splitting the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_SIZE//10)
        documents = text_splitter.split_documents(raw_documents)
        print(f"   Built {len(documents)} text chunks.")
        # building the database
        database = FAISS.from_documents(documents, EMBEDDING_MODEL)
        database.save_local(self.persona_path)
        print(f"   Built vector database.")
        # returning the result
        print(f"   Done.")
        return database

    def get_sources(self, user_name, chat_history, nb_sources=4, separator='=========', verbose=False):
        """gets text extracts that seem relevant to the question"""
        # gets a summary of the themes in the question to help with embedding search
        embeding_text = embedding_chain({'user_name':user_name, 'name':self.name, 'chat_history': chat_history})['text'].strip()
        if verbose: print(f"> Embeding text: {embeding_text}")
        # gets documents to help answer the question
        documents = self.database.similarity_search(embeding_text, k=nb_sources)
        # converts them into strings
        sources_by = ""
        sources_about = ""
        for doc in documents:
            if doc.metadata['written_by_persona']:
                sources_by += doc.page_content
                sources_by += f"\n{separator}\n"
            else:
                sources_about += doc.page_content
                sources_about += f"\n{separator}\n"
        # merge the strings
        sources = ""
        if sources_by != "":
            sources += f"WRITTEN BY {self.name.upper()}:\n{sources_by}"
        if sources_about != "":
            if sources != "": sources += '\n'
            sources += f"WRITTEN ABOUT {self.name.upper()}:\n{sources_about}"
        if verbose: print(sources)
        return sources

    # ----- CHAT -----

    def chat(self, user_name, chat_history, sources=None, nb_sources=4, verbose=False):
        """
        pass a user_name, chat_history (string) and optionaly sources to the persona
        returns a pair (answer, sources)
        """
        # gets sources if needed
        if sources is None:
            sources = self.get_sources(user_name, chat_history, nb_sources, verbose=verbose)
        # gets an answer from the model and returns
        if self.is_strict:
            answer = strict_answering_chain({'name':self.name, 'sources': sources, 'chat_history': chat_history})['text'].strip()
        else:
            answer = answering_chain({'name':self.name, 'sources': sources, 'chat_history': chat_history})['text'].strip()
        return (answer, sources)

    def check_answer(self, answer, sources):
        """
        factcheck the chatbot's answer given the corresponding sources
        """
        return check_chain({'name':self.name, 'sources': sources, 'answer':answer})['text'].strip()
