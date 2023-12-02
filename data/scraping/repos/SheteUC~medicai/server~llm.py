import os
import re
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
from langchain.llms.cohere import Cohere
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import LLMChain
from langchain.embeddings import CohereEmbeddings
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model


from scholar import Scholar

load_dotenv()

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)


def init_get_abstract_chain(llm, verbose=False):
    template = """
    You are a superintelligent AI trained for medical situations specifically.
    You task is to receive the 2000 tokens of a medical research PDF in plaintext 
    and summarize it the whole thing based on the abstract of the paper.
    Your summary should only be at most two sentences long. Do not include anything other
    than your summary. The summary shuld caters to the patient demographic of {demographics}.
    Here is the text:
    {section}
    Your summary is:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["section", "demographics"],
    )

    return LLMChain(prompt=prompt, llm=llm, verbose=verbose)

def cosine_similarity(text, keywords):
    text_embedding = embeddings.embed_query(text)
    total = 0
    for keyword in keywords:
        keyword_embedding = embeddings.embed_query(keyword)
        similarity = 1 - cosine(text_embedding, keyword_embedding)
        total += similarity
    return total / len(keywords)
    

def init_all_chain(verbose=False):
    llm_dict = {
        "get_abstract": OpenAI(openai_api_key=OPENAI_API_KEY),
    }
    
    get_abstract_chain = init_get_abstract_chain(llm=llm_dict["get_abstract"], verbose=verbose)

    return {
        "get_abstract": get_abstract_chain
    }

def get_abstract(llm_chain, section, bio_information):
    if llm_chain:
        return llm_chain.run({
            "section": section,
            "demographics": bio_information,
        })
    raise ValueError()

def summarize_and_relevancy(llm_dict, bio_information, text):
    summary = get_abstract(llm_chain=llm_dict["get_abstract"], section=text, bio_information=bio_information)
    similarity = cosine_similarity(text=summary, keywords=bio_information)
    return summary, round(similarity, 2)

def text_chunker(text, chunk_size=2000):
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    return text_splitter.split_text(text)

def get_abstract_text_chunk(text, chunk_size=2000):
    abstract = re.search(r"abstract", text, re.IGNORECASE)
    if abstract:
        text = text[abstract.start():]
        text_chunked = text_chunker(text=text, chunk_size=chunk_size)[0]
        return text_chunked
    
    return text_chunker(text=text, chunk_size=chunk_size)[0]


if __name__ == "__main__":
    demographics = ["african american", "teen", "depression"]
    chain_dict = init_all_chain(verbose=True)
    papers = Scholar.get_all_papers("Polycystic Ovary Syndrome", demographics=demographics)
    paper = papers[0]
    # get the fist 2000 tokens after the first occurence of "abstract"
    text_chunked = get_abstract_text_chunk(text=paper["text"], chunk_size=500)
    abstract, similarity = summarize_and_relevancy(llm_dict=chain_dict, bio_information=demographics, text=text_chunked)
    print(abstract)
    print(similarity)
