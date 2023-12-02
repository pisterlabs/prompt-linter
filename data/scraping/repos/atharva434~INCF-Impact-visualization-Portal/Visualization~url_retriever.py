from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import cohere
from langchain.llms import Cohere
import nltk
import json
from .serp import stats_finder
from langchain.text_splitter import CharacterTextSplitter
llm = Cohere(cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi",temperature=0)
# import nltk
# nltk.download('averaged_perceptron_tagger')

def diseases(tex):
    prompt=f"write the mental disorders mentioned in {tex} if no mental disorders present say 'no disease found'"
    diseases=llm(prompt)
    nltk_tokens = nltk.word_tokenize(diseases)
    return nltk_tokens
    # print(nltk_tokens)

def impact_count(disease):
    # prompt2=f"tell me number of people suffering from {disease} no need to give background information"
    # return llm(prompt2)
    return stats_finder(disease)

def research_params(tex):
    prompt=f"""Extract the aim,use of project, and the real world impact of this project and expand in detail on the impact {tex} 
 the project description is {tex} give in json format with keys (aim use and impact)"""
    params=llm(prompt)
    print(params)
    params=json.loads(params)
    print(params["impact"])
    return params
    

# prompt_check=f"write 'yes' if any mental disorder is detected in {tex} otherwise return 'no'"
# print(llm(prompt_check))

def impact(url):
    urls = [url]
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    tex=str(docs[0])
    list_diseases=diseases(tex)
    print(list_diseases)
    
    if "no"not in list_diseases:
        disease_stats=""
        l=[]
        for i in list_diseases:
            
            link,stat=impact_count(i)
            d={"disease":i,"source":link,"impact":stat}
            l.append(d)
            # disease_stats=disease_stats+stat
        return l,"disease"
    else:
        params=research_params(tex)
        return params,"research"

def ingest(urls):
    loader = UnstructuredURLLoader(urls=urls) #need to change
    # Split pages from pdf
    pages = loader.load()

    #  to store it in a folder name titan
    persist_directory = 'test'
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(pages)
    embeddings = CohereEmbeddings(cohere_api_key="4aJ9yWbIrOzI2W5LZeLeIdin2AYMpkq18PffLuvi")
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None
    return True


def calculator(impact):
    s=" ".join(impact)
    prompt="""Identify the mental disorders mentioned in""" + s+"""and tell me all the unique diseases along with their count in 
     in the dictionary format with key and value as string
    {"disease":"count"}
    if no mental disorders present say 'no disease found'
    """
    diseases=llm(prompt)
    print(diseases)
    diseases=json.loads(diseases)
    return diseases

def project_details(title,url):
    urls=[url]
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    tex=str(docs[0])
    prompt=f"""Give me a structured output in json format covering the description,the tech stack, the domain,the subdomain
using {tex} for information"""
    params=llm(prompt)
    params=json.loads(params)
    return params

def clean_with_llm(impact,org):
    prompt=f"Convert {impact} into a proper impactful and concise paragraph without losing any piece of information present in text and add a line that {org} specifically is helping those people who suffer"
    params=llm(prompt)
    return params


        