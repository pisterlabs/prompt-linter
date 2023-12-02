from pathlib import Path as p

from langchain import OpenAI
from langchain.llms import VertexAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

import urllib
    
PROJECT_ID = "foret-399300"
LOCATION = "us-east1" #e.g. us-central1

import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION)

data_folder = p.cwd() / "data"
p(data_folder).mkdir(parents=True, exist_ok=True)

tesla_pdf = "https://www.tesla.com/ns_videos/2022-tesla-impact-report-highlights.pdf"


def summarize(pdf_url):

    pdf_file = str(p(data_folder, pdf_url.split("/")[-1]))

    urllib.request.urlretrieve(pdf_url, pdf_file)


    pdf_loader = PyPDFLoader(pdf_file)
    print(pdf_loader)
    pages = pdf_loader.load_and_split()
    text = "\n".join([(page if isinstance(page, str) else page.page_content) for page in pages])


    llm = VertexAI(model_name="text-bison@001")


    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], 
    chunk_size=3000, chunk_overlap=100)

    docs = text_splitter.create_documents([text])

    map_prompt_template = """
                    Take 2-3 bullet points from the following text delimited by triple backquotes that includes the most important details and facts for investors and environmentalists to consider. Please consider the following information if it is present in the passage:
            Key risks in investing in the company, Environmental positives and negatives, Social positives and negatives, Governance positives and negatives, Adherance to government regulations. Also, focus on extracting sentences with these keywords delimited by parentheses: (ESG, Sustainability, Environment, Diversity, Climate, Equality, Carbon, Conscious, Responsibility, CSR, Environment, Social, and Governance, Green, Renewable, Recycle, Discrimination, Racism, Sexism).
                    ```{text}```
                    """

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
                    Write a 10-20 bullet point summary of the following text delimited by triple backquotes.the most important details and facts for investors and environmentalists to consider. Please consider the following information if it is present in the passage:
            Key risks in investing in the company, Environmental positives and negatives, Social positives and negatives, Governance positives and negatives, Adherance to government regulations. Also, focus on extracting sentences with these keywords delimited by parentheses: (ESG, Sustainability, Environment, Diversity, Climate, Equality, Carbon, Conscious, Responsibility, CSR, Environment, Social, and Governance, Green, Renewable, Recycle, Discrimination, Racism, Sexism).
                    ```{text}```
                    BULLET POINT SUMMARY:
                    """

    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text"]
    )

    summary_chain = load_summarize_chain(llm=llm,
    chain_type='map_reduce',
    map_prompt=map_prompt,
    combine_prompt=combine_prompt,
    )

    summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce')
    output = summary_chain.run(docs)
    return(output)

print(summarize(tesla_pdf))

