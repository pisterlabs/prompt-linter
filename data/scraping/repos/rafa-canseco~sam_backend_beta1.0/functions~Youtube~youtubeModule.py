from langchain.callbacks import get_openai_callback
from langchain.document_loaders import YoutubeLoader
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
import ssl
import os
from decouple import config
import openai

os.environ["OPENAI_API_KEY"] =config("OPEN_AI_KEY")
openai.api_key = config("OPEN_AI_KEY")

url_esp = "https://www.youtube.com/watch?v=Ok9qHeLxu10"
url_ingles = "https://www.youtube.com/watch?v=DWUdGhRrv2c"
question = "de que trata el video?"
selection="normal"

def preguntar_youtube(url,question):
    with get_openai_callback() as cb:
        ssl._create_default_https_context = ssl._create_stdlib_context
        loader = YoutubeLoader.from_youtube_url(url,
        language=["en","es"],
        translation="es")
        result = loader.load()
        llm =OpenAI(temperature=0,openai_api_key=openai.api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=0)
        texts=text_splitter.split_documents(result)
        embeddings = OpenAIEmbeddings()
        db =Chroma.from_documents(texts,embeddings)
        retriever =db.as_retriever(search_type="similarity",search_kwargs={"k":2})
        qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0),chain_type="stuff",retriever=retriever,return_source_documents=False)
        query=question
        result=qa({"query":query})
        print(result)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return result
    
def youtube_resume(url,selection):
    with get_openai_callback() as cb:
        ssl._create_default_https_context = ssl._create_stdlib_context
        loader = YoutubeLoader.from_youtube_url(url,
          language=["en","es"],
        translation="es")
        text=loader.load()
        llm =OpenAI(temperature=0,openai_api_key=openai.api_key)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=0)
        texts=text_splitter.split_documents(text)
        
        if (selection == "normal"):
            summary_chain = load_summarize_chain(llm=llm,chain_type="map_reduce",verbose=False)
            resume =summary_chain.run(texts)  
            print("---------------------------")
            print("Resumen:   ")
            print(resume)
            print("---------------------------")
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Successful Requests: {cb.successful_requests}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            return resume
        
        elif (selection == "extended"):

            map_prompt = """
            Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:
            """
            map_prompt_template = PromptTemplate(template=map_prompt,input_variables=["text"])
            combine_prompt = """
            Write a concise summary of the following text delimited by triple backquotes.
            Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
            Your response should be at least three paragraphs and fully encompass what was said in the passage.
            Always deliver your response in spanish. 
                    ```{text}```
            FULL SUMMARY:
            """
            combine_prompt_template = PromptTemplate(template=combine_prompt,input_variables=["text"])
            summary_chain = load_summarize_chain(llm=llm,chain_type="map_reduce",
                                                 verbose=True,
                                                 map_prompt=map_prompt_template,
                                                 combine_prompt=combine_prompt_template)
            resume = summary_chain.run(texts)
            print("resumen extended")
            print(resume)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Successful Requests: {cb.successful_requests}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            return resume
        
        elif (selection == "bulletpoints"):
            map_prompt = """
            Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:
            """
            map_prompt_template = PromptTemplate(template=map_prompt,input_variables=["text"])
            combine_prompt = """
            Write a concise summary of the following text delimited by triple backquotes.
            Return your response in bullet points which covers the key points of the text.
            Always deliver your response in spanish.
            ```{text}```
            BULLET POINT SUMMARY:
            """
            combine_prompt_template = PromptTemplate(template=combine_prompt,input_variables=["text"])
            summary_chain = load_summarize_chain(llm=llm,chain_type="map_reduce",
                                                 verbose=True,
                                                 map_prompt=map_prompt_template,
                                                 combine_prompt=combine_prompt_template)
            resume = summary_chain.run(texts)
            print("resumen bulletpoints")
            print(resume)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Successful Requests: {cb.successful_requests}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            return resume


# preguntar_youtube(url_ingles,question)       
# youtube_resume(url_esp,selection)
        
   