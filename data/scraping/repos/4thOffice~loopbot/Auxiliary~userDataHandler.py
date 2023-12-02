import sys
sys.path.append('./Auxiliary')
import Auxiliary.databaseHandler as databaseHandler
import Auxiliary.loader as loader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

class UserDataHandler:
    def __init__(self):
        underlying_embeddings = OpenAIEmbeddings()
        self.user_data = {}
        self.fs = LocalFileStore("./cache/")

        userIDS = databaseHandler.get_unique_user_ids()
        for userID in userIDS:
            self.user_data[userID] = {}
            json_data = databaseHandler.get_user_json_data(userID)
            
            for file_name in json_data:
                cached_user_embedder = CacheBackedEmbeddings.from_bytes_store(
                    underlying_embeddings, self.fs, namespace=userID + "-" + file_name
                )
                if("CUSTOM_" not in file_name):
                    
                    loader_ = loader.JSONLoader(file_path="")
                    documents = loader_.loadResponses(json_data[file_name])
                    self.user_data[userID][file_name] = {"docs": FAISS.from_documents(documents, cached_user_embedder), "json": json_data[file_name]}
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        # Set a really small chunk size, just to show.
                        chunk_size = 300,
                        chunk_overlap  = 50,
                        length_function = len,
                        add_start_index = True,
                    )
                    documents = text_splitter.create_documents([json_data[file_name]])

                    if "other" in self.user_data[userID]:
                        self.user_data[userID]["other"]["docs"].add_documents(documents)
                    else:
                        self.user_data[userID]["other"] = {"docs": FAISS.from_documents(documents, cached_user_embedder)}
        print(self.user_data)

    def checkUserData(self, userID):
        if userID not in self.user_data:
            self.user_data[userID] = {}
            databaseHandler.add_user_json_data(userID, "good_responses")
            databaseHandler.add_user_json_data(userID, "bad_responses")
            databaseHandler.add_user_json_data(userID, "good_responses_email")
            databaseHandler.add_user_json_data(userID, "bad_responses_email")

            json_data = databaseHandler.get_user_json_data(userID)
            
            underlying_embeddings = OpenAIEmbeddings()

            for file_name in json_data:
                cached_embedder_good = CacheBackedEmbeddings.from_bytes_store(
                        underlying_embeddings, self.fs, namespace=userID + "-" + file_name
                )
                if(file_name != "other"):
                    loader_ = loader.JSONLoader(file_path="")
                    documents = loader_.loadResponses(json_data[file_name])
                    
                    self.user_data[userID][file_name] = {"docs": FAISS.from_documents(documents, cached_embedder_good), "json": json_data[file_name]}

                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        # Set a really small chunk size, just to show.
                        chunk_size = 300,
                        chunk_overlap  = 50,
                        length_function = len,
                        add_start_index = True,
                    )
                    documents = text_splitter.create_documents([json_data[file_name]])

                    if "other" in self.user_data[userID]:
                        self.user_data[userID]["other"]["docs"].add_documents(documents)
                    else:
                        self.user_data[userID]["other"] = {"docs": FAISS.from_documents(documents, cached_embedder_good)}



    def addToUserDocument(self, userID, document_content, document_name):
        underlying_embeddings = OpenAIEmbeddings()
        document_name = document_name.replace(" ", "")
        cached_embedder_good = CacheBackedEmbeddings.from_bytes_store(
                underlying_embeddings, self.fs, namespace=userID + "-" + document_name
        )

        databaseHandler.insert_json_data(userID, ("CUSTOM_" + document_name), document_content)

        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 300,
            chunk_overlap  = 50,
            length_function = len,
            add_start_index = True,
        )
        
        documents = text_splitter.create_documents([document_content])
        print("other" in self.user_data[userID])
        if "other" in self.user_data[userID]:
            print("text 2", self.user_data[userID]["other"])
            self.user_data[userID]["other"]["docs"].add_documents(documents)
            print("text 3", self.user_data[userID]["other"])
        else:
            print("text 4", self.user_data[userID])
            self.user_data[userID]["other"] = {"docs": FAISS.from_documents(documents, cached_embedder_good)}
            print("text 5", self.user_data[userID]["other"])
        
        print("all docs ", self.user_data[userID]["other"]["docs"])