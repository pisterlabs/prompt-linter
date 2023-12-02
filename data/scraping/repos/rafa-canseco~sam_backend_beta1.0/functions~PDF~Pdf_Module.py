import os
from decouple import config
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
from sklearn.cluster import KMeans
import time

os.environ["OPENAI_API_KEY"] = config("OPEN_AI_KEY")
openai.api_key = config("OPEN_AI_KEY")

llm = OpenAI(temperature=0, openai_api_key=openai.api_key)
user = "test"
selection1 = "normal" 
selection2 = "extended" 
selection3 = "bulletpoints" 


def small_Archive(url,selection):
    with get_openai_callback() as cb:
  
        loader = PyPDFLoader(url)
        pages = loader.load()
        pages = pages[3:]

        text = ""
        for page in pages:
            text += page.page_content
        
        text = text.replace('\t', ' ')

        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=8000,chunk_overlap=500)
        docs = text_splitter.create_documents([text])
        num_docs = len(docs)
        num_tokens_first_doc = llm.get_num_tokens(docs[0].page_content)
        print (f"Now we have {num_docs} documents and the first one has {num_tokens_first_doc} tokens") 

        if (selection == "normal"):

            summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce',verbose=False)
            output = summary_chain.run(docs)
            print("resumen normal")

        elif (selection == "extended"):
        
            map_prompt = """
            Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:
            """
            map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
            combine_prompt = """
            Write a concise summary of the following text delimited by triple backquotes.
            Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
            Your response should be at least 1600 characters and fully encompass what was said in the passage.
            Always deliver your response in spanish.
            
                    ```{text}```
            FULL SUMMARY:
            """
            combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
            summary_chain = load_summarize_chain(llm=llm,
                                        chain_type='map_reduce',
                                        map_prompt=map_prompt_template,
                                        combine_prompt=combine_prompt_template,
                                    verbose=True
                                        )
            output = summary_chain.run(docs)
            print("resumen extended")


        elif (selection == "bulletpoints"):
            map_prompt = """
            Write a concise summary of the following:
            "{text}"
            CONCISE SUMMARY:
            """
            map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
            combine_prompt = """
            Write a concise summary of the following text delimited by triple backquotes.
            Return your response in bullet points which covers the key points of the text.
            Always deliver your response in spanish.
            ```{text}```
            BULLET POINT SUMMARY:
            """
            combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
            summary_chain = load_summarize_chain(llm=llm,
                                        chain_type='map_reduce',
                                        map_prompt=map_prompt_template,
                                        combine_prompt=combine_prompt_template,
                                    verbose=True
                                        )
            output = summary_chain.run(docs)
            print("bulletpoints")


        print(output)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        return output


def big_archive(url,selection):
    inicio = time.time()
    with get_openai_callback() as cb:
        print("big archive")
        loader = PyPDFLoader(url)
        pages = loader.load()
        pages = pages[3:]

        text = ""
        for page in pages:
            text += page.page_content
        
        text = text.replace('\t', ' ')
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=8000, chunk_overlap=5000)
        docs = text_splitter.create_documents([text])
        embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

        vectors = embeddings.embed_documents([x.page_content for x in docs])

        num_clusters = 7
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

        closest_indices = []

        for i in range(num_clusters):
            
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            
            closest_index = np.argmin(distances)
            
            closest_indices.append(closest_index)
        
        selected_indices = sorted(closest_indices)

        llm3 = ChatOpenAI(temperature=0,
                 openai_api_key=openai.api_key,
                 max_tokens=1000,
                 model='gpt-3.5-turbo'
                )
        map_prompt = """
        You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
        Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
        Your response should be at least three paragraphs and fully encompass what was said in the passage.
        Always deliver your response in spanish.
        ```{text}```
        FULL SUMMARY:
        """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        map_chain = load_summarize_chain(llm=llm3,
                             chain_type="stuff",
                             prompt=map_prompt_template)
        selected_docs = [docs[doc] for doc in selected_indices]
        # Make an empty list to hold your summaries
        summary_list = []

        # Loop through a range of the lenght of your selected docs
        for i, doc in enumerate(selected_docs):
            
            # Go get a summary of the chunk
            chunk_summary = map_chain.run([doc])
            
            # Append that summary to your list
            summary_list.append(chunk_summary)
            
            print (f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")
        
        summaries = "\n".join(summary_list)

        # Convert it back to a document
        summaries = Document(page_content=summaries)
        print (f"Your total summary has {llm.get_num_tokens(summaries.page_content)} tokens")

        llm4 = ChatOpenAI(temperature=0,
                 openai_api_key=openai.api_key,
                 max_tokens=3000,
                 model='gpt-4',
                 request_timeout=120
                )
        if (selection == "normal"):
            combine_prompt = """
            You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
            Your goal is to give a short summary of what happened in the story.
            The reader should be able to grasp what happened in the book in a concise way.
            Always deliver your response in spanish.

            ```{text}```
            SHORT SUMMARY:
            """
            combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
            reduce_chain = load_summarize_chain(llm=llm4,
                                chain_type="stuff",
                                prompt=combine_prompt_template,
    #                              verbose=True # Set this to true if you want to see the inner workings
                                    )
            output = reduce_chain.run([summaries])
            print (output)
        elif( selection =="extended"):
            combine_prompt = """
            You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
            Your goal is to give a verbose summary of what happened in the story.
            The reader should be able to grasp what happened in the book.
            Always deliver your response in spanish.

            ```{text}```
            VERBOSE SUMMARY:
            """
            combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
            reduce_chain = load_summarize_chain(llm=llm4,
                                chain_type="stuff",
                                prompt=combine_prompt_template,
    #                              verbose=True # Set this to true if you want to see the inner workings
                                    )
            output = reduce_chain.run([summaries])
            print (output)
        elif( selection =="bulletpoints"):
            combine_prompt = """
            You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```)
            Your goal is to give a bulletpoints summary of what happened in the story.
            The reader should be able to grasp what happened in the book.
            Return your response in bullet points which covers the key points of the text
            Always deliver your response in spanish.

            ```{text}```
            BULLETPOINTS SUMMARY:
            """
            combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
            reduce_chain = load_summarize_chain(llm=llm4,
                                chain_type="stuff",
                                prompt=combine_prompt_template,
    #                              verbose=True # Set this to true if you want to see the inner workings
                                    )
            output = reduce_chain.run([summaries])
            print (output)
            fin = time.time()
            tiempo_total = fin-inicio
            print(tiempo_total)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Successful Requests: {cb.successful_requests}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            return output




