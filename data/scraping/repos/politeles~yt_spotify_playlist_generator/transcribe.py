import os
import argparse
import datetime
import json
from typing import Iterable
import openai
from dotenv import load_dotenv,find_dotenv
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.chains.retrieval_qa import BaseRetrievalQA
import file_json as fj




# parse youtube video using langchain and openai whisper parser to gererate a document
def parse_youtube_video(video_id:str,save_dir:str,transcript_dir:str)->Document:
    baseURL = "https://www.youtube.com/watch?v="
    loader = YoutubeAudioLoader([baseURL+video_id],save_dir)
    # check that the file exists at transcript_dir and load it
    if transcript_dir != None:
        if os.path.isfile(transcript_dir+"/"+video_id+".json"):
            print("trying to load transcript from file...")
            document = GenericLoader.from_filesystem(transcript_dir+"/"+video_id+".json").load()
    else:
        parser = OpenAIWhisperParser()
        document = GenericLoader(loader, parser).load()
    return document

def split_documents(docs:Iterable[Document])->Iterable[Document]:
    # split documents into sentences
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
    )
    splits = text_splitter.split_documents(docs)
    return splits

# generate embedings in chroma format
def generate_embeddings(splits:list[Document],removeChroma:bool=False,persist_directory:str="chroma")->Chroma:
    embedding = OpenAIEmbeddings()
    if removeChroma:
        os.rmdir(persist_directory)
    if splits == None:
        vectordb = Chroma(persist_directory=persist_directory,embedding_function=embedding)
    else:
        vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
        )
    return vectordb

def get_llm()->ChatOpenAI:
    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        llm_name = "gpt-3.5-turbo-0301"
    else:
        llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    return llm


# generate prompt template
def generate_prompt_template(vectordb:Chroma,question:str)-> BaseRetrievalQA:
    template = """Usa las siguientes piezas de contexto para responder las preguntas del final. Si no sabes la respuesta, dí que no sabes, no trates de inventar una respuesta. Usa una lista en formato json para generar la respuesta.
    {context}
    Pregunta: {question}
    Respuesta:"""
    qa_chain_prompt = PromptTemplate.from_template(template)
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_chain_prompt}
    )

    result = qa_chain({"query": question})
    return result




def main():
    parser = argparse.ArgumentParser(description='Transcribe videos from youtube and get the list of songs')
    parser.add_argument('file_name', help='json file with video titles and ids')
    parser.add_argument('video_id', help='id of the video to be parsed')
    parser.add_argument('save_dir', help='directory to save the audio file')
    parser.add_argument('transcript_dir', help='directory to save the transcription')
    args = parser.parse_args()

    file_name = args.file_name
    video_id = args.video_id
    save_dir = args.save_dir
    transcript_dir = args.transcript_dir

    if save_dir == None:
        save_dir = os.getcwd()
    print("loading environment variables...")
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # print("Downloading and listening to the video...")
    # document = parse_youtube_video(video_id,save_dir,transcript_dir=transcript_dir)
    # print(document)
    # splits = split_documents(document)
    splits = None
    vectordb = generate_embeddings(splits)
    question = "Listado de discos o canciones mencionadas en el vídeo, con una breve reseña en el formato descripción, añade todos los discos se que mencionen con el siguiente formato y deja vacios los campos que no conozcas {'album','song','artist','year','description'}"
    result = generate_prompt_template(vectordb,question)
    # parse result["result] to json
    res = json.loads(result["result"])

    # save result to json file
    fj.write_json(res,video_id+".json")

    print(res)



if __name__ == "__main__":
    main()
