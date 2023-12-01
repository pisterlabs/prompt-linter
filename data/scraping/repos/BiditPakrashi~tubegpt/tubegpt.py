
from typing import List
from audio.simple_audio_info_extractor import SimpleAudioInfoExtractor
#from audio.simple_audio_info_extractor import SimpleAudioInfoExtractor
from chain.llm_chain import LLMChain
from chain.retrieval_qa_chain import RetrievalQAChain
from embedding.chromadb import ChromaDB
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from prompt.model_prompt import ModelPrompt
from video.video_description_extractor import VideoDescriptionExtractor
from video.simple_video_information_extractor import SimpleVideoInfoExtractor
import os
import openai  
from langchain.retrievers.merger_retriever import MergerRetriever     
from dotenv import load_dotenv, find_dotenv
import re

# Environment Variables
TIME_WINDOW = 10 # data for audio and vidio will be consolidated in this time (in secs)
CAPTION_PER_IMAGE =  3 #for LAVIS blip 2 model
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

class TubeGPT():
    def __init__(self,vid_url,save_dir,db_dir="./db") -> None:
        self.vid_url:str = vid_url
        self.save_dir:str = save_dir
        self.audio_db:ChromaDB = None
        self.vision_db:ChromaDB = None
        self.audio_chain:RetrievalQAChain = None
        self.vision_description_chain:RetrievalQAChain = None
        self.vision_chain:RetrievalQAChain = None
        self.db_dir = db_dir
        self.video_disc_extractor:VideoDescriptionExtractor = VideoDescriptionExtractor()
        self.merger_retreiver:MergerRetriever = None
        self.merged_chain:RetrievalQAChain = None




    def process_audio(self,vid_url, save_dir,audio_extractor=SimpleAudioInfoExtractor(), embeddings=OpenAIEmbeddings()):
        '''
        processes audio to first get captions by using audio parser like openaiwhisper extractor provided
        then and then the documents are saved as embeddings in chromadb
        '''  
        self.__extract_folder_id()
        docs = audio_extractor.audio_to_text(vid_url,save_dir=f"{save_dir}/{self.folder_id}")
        self.__save_audio_embeddings(docs,embeddings=embeddings,db_path=f"{self.db_dir}/{self.folder_id}/audio")

    # save audio embeddings
    def __save_audio_embeddings(self,docs,embeddings, db_path):
        if not (docs):
            print("docs are empty")
            return
        self.audio_db = ChromaDB(db_path)
        self.audio_db.save_embeddings(docs,embeddings)

    def __extract_folder_id(self):
        print(self.vid_url)
        folder_search =  re.search("v=(.*?)(&|$)",self.vid_url[0])
        if(folder_search):
            self.folder_id = folder_search.group(1)
            print(f"extracted folder id:{self.folder_id}")

    def load_audio(self,db_path, embeddings = OpenAIEmbeddings()):
        self.audio_db = ChromaDB(db_path)
        self.audio_db.load_db(embedding_function=embeddings)
        print(f"loaded chroma audio db from {db_path}")

    def load_video(self,db_path, embeddings = OpenAIEmbeddings()):
        self.video_db = ChromaDB(db_path)
        self.video_db.load_db(embedding_function=embeddings)
        print(f"loaded chroma video db from {db_path}")


    #2. Query audio
    def query_audio(self,question:str,reader)->str:
        '''
        queries using audio reader retrieval model and returns the response
        ''' 
        print(f"question: {question}")
        print(f"reader:{reader}")

        if(self.audio_chain):
            print("audio chain already exists, querying using that")
            chain = self.audio_chain
        else:
            chain = self.__get_audio_chain(self.audio_db,reader=reader)
        print("audio chain ->")
        print(chain)
        response = chain.query(question)
        print(response)
        return response['result']
    
    #2. Query audio with loading existing db
    def query_audio_withload(self,question:str,reader,audio_db)->str:
        '''
        queries using audio reader retrieval model and returns the response
        ''' 

        print(f"question: {question}")
        print(f"reader:{reader}")
        if(self.audio_chain):
            print("audio chain already exists, querying using that")
            return self.audio_chain.query(question)
        
        chain = self.__get_audio_chain(audio_db,reader=reader)
        print(self.audio_chain)
        return chain.query(question)
    
    
    def __get_audio_chain(self, retreiverdb, reader) -> RetrievalQAChain:
        if(self.audio_chain):
            return self.audio_chain
        
        print("creating new audio chain")
        self.audio_chain = RetrievalQAChain()
        prompt = ModelPrompt()
        print(
            prompt.get_audio_prompt().format(
                    context="A youtube user asking about contents ",
                    question="Why do we need to zero out the gradient before backprop at each step?")
)
        ret = retreiverdb.db.as_retriever(search_kwargs={"k": 1})
        self.audio_chain.create_chain(retreiver=ret,reader=reader,prompt=prompt.get_audio_prompt())
        return self.audio_chain
    

    def process_vision(self,file_path, video_extractor=SimpleVideoInfoExtractor(1,3),embeddings=OpenAIEmbeddings()):
        '''
        processes video to first get captions using a vision chain by using the video info extractor provided
        then the captions are converted to descriptions using an llm chain and then the documents
        are saved as embeddings in chromadb
        '''   
        docs = video_extractor.video_to_text(self.vid_url,file_path)
        description_docs = self.__caption_to_description(docs) #convert docs to description
        self.__save_video_embeddings(description_docs,embeddings,f"{self.db_dir}/{self.folder_id}/video")

    def process_vision_from_desc(self,file_path,embeddings=OpenAIEmbeddings()):
        '''
        processes video to first get captions using a vision chain by using the video info extractor provided
        then the captions are converted to descriptions using an llm chain and then the documents
        are saved as embeddings in chromadb
        '''   
        description_docs = self.__get_vision_description_docs(file_path=file_path)
        self.__save_video_embeddings(description_docs,embeddings,f"{self.db_dir}/{self.folder_id}/video")


    def __caption_to_description(self,docs_vision_caption)->List:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        splits = text_splitter.split_documents(docs_vision_caption)

        processed_splits = []
        for split in splits:
            result = self.__get_video_description_chain()({"image_captions": split}, return_only_outputs=True)
            processed_splits.append(result['text'])

        return processed_splits


    # save video embeddings
    def __save_video_embeddings(self,docs,embeddings, db_path):
        if not (docs):
            print("docs are empty")
            return
        self.vision_db = ChromaDB(db_path)
        self.vision_db.save_embeddings(docs,embeddings)


    def __get_video_description_chain(self, retreiverdb, reader) -> RetrievalQAChain:
        if(self.vision_description_chain):
            return self.vision_description_chain
        
        self.vision_description_chain = LLMChain()
        prompt = ModelPrompt()
        self.vision_description_chain.create_chain(llm=ChatOpenAI(temperature=0),prompt=prompt.get_text_description_prompt())
        return self.vision_description_chain
    
    #2. Query vision
    def query_vision(self,question:str,reader)->str:
        '''
        queries using vision reader retrieval model and returns the response
        ''' 
        if(self.vision_chain):
            chain = self.vision_chain
        else:
            chain = self.__get_vision_chain(self.vision_db,reader=reader)
        print("vision chain ->")
        print(chain)
        response = chain.query(question)
        print(response)
        return response['result']
    
    def __get_vision_chain(self, retreiverdb, reader) -> RetrievalQAChain:
        if(self.vision_chain):
            return self.vision_chain
        
        self.vision_chain = RetrievalQAChain()
        prompt = ModelPrompt()
        self.vision_chain.create_chain(retreiver=retreiverdb.db.as_retriever(search_kwargs={"k": 1}),reader=reader,prompt=prompt.get_video_prompt())
        return self.vision_chain
    
    def __get_vision_description_docs(self,file_path):
        if not file_path:
            return "file path is empty"
        
        return self.video_disc_extractor.load_from_file(file_path)
    
    def merge_retreivers(self,list_retreivers:List):
        self.merger_retreiver = MergerRetriever(retrievers=list_retreivers)

    #2. Query merged
    def query_merged(self,question:str,reader)->str:
        '''
        queries using audio and vision reader retrieval model and returns the response
        ''' 
        if(self.merged_chain):
            chain = self.merged_chain
        else:
            chain = self.__get_merged_chain(self.merger_retreiver,reader=reader)
        print("merged chain ->")
        print(chain)
        response = chain.query(question)
        print(response)
        return response['result']

    
    def __get_merged_chain(self, retreiver, reader) -> RetrievalQAChain:
        if(self.merged_chain):
            return self.merged_chain
        
        self.merged_chain = RetrievalQAChain()
        prompt = ModelPrompt()
        self.merged_chain.create_chain(retreiver=retreiver,reader=reader,prompt=prompt.get_video_prompt())
        return self.merged_chain
        



