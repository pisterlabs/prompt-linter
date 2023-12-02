import os
from dotenv import load_dotenv
# os.environ["OPENAI_API_KEY"]

from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain.agents.tools import Tool
from langchain import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain, create_extraction_chain_pydantic
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from .githubanalyzer import GithubProfileAnalyzer
import os
import json
import requests
# env_path = os.path.join(os.getcwd() , ".env")

load_dotenv(".env")
openapi_key = os.getenv('OPENAI_API_KEY')
serper_api_key = os.getenv("SERP_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
print(anthropic_api_key)


from typing import Optional, List
from pydantic import BaseModel, Field , validator

class Applicant(BaseModel):
    Name: str
    Contact: str
    Location: str
    University: str
    Degree: str
    Major: str
    Graduation_Year: int
    Skills: list[str]
    Certifications: Optional[list[str]]
    Languages: Optional[list[str]]
    Role: str
    Company: str
    Start_Date: str
    End_Date: str
    Experience: int
    Responsibilities: list[str]
    Achievements: Optional[list[str]]
    Projects: Optional[list[str]]
    LinkedIn_Url: Optional[str] = Field( None, description="The applicant's LinkedIn profile URL. If a proper URL is not available, please format it correctly.")
    # Add description for Github_Url
    Github_Url: Optional[str] = Field( None, description="The applicant's GitHub profile URL. If a proper URL is not available, please format it correctly.")

    Hobbies: Optional[list[str]]

    @validator("Github_Url", "LinkedIn_Url", pre=True, always=True)
    def format_url(cls, v):
        if v and not v.startswith(("http://", "https://")):
            return f"https://{v}"
        return v



from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

from account.models import JobInsightData
from typing import List

from langchain.document_loaders import PyPDFLoader




from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA




def get_insights(data , query):

    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
    )

    texts = text_splitter.create_documents([data])

    embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_documents(texts, embeddings)

    # docs = db.similarity_search(query)

    llm = ChatOpenAI(temperature=0.0)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 3}))

    output = qa.run(query)
    print(output)


    return output







def analyze_github_profile(profile_url):
    load_dotenv(".env")

    analyzer = GithubProfileAnalyzer(anthropic_api_key=anthropic_api_key, serper_api_key=serper_api_key)
    return analyzer.analyze(profile_url)


def extract_resume_information(filename , job):
    pdf_path = os.path.join(os.getcwd(), "media", "resumes", filename)
    # Load PDF
    loader = PyPDFLoader(pdf_path)

    # media/resumes/Poojan_vig_resume.pdf
    final = loader.load()
    resume = final[0].page_content
    llm = ChatOpenAI(temperature=0.0)

    pydantic_parser = PydanticOutputParser(pydantic_object=Applicant)
    
    format_instructions = pydantic_parser.get_format_instructions()

    template_string = """ You are a professional HR that and information extractor that can analyze the resume of the person and extract the contents as given in the instructions below
    Resume: ```{resume}```
    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_template(template=template_string)
    messages = prompt.format_messages(resume=resume, format_instructions=format_instructions)

    output = llm(messages)

    output = output.content

    json_out = json.loads(output)

    git_out = None

    # print(json_out)

    if json_out['Github_Url']:
        print("Inside github agent " , json_out['Github_Url'])
        git_out = analyze_github_profile(json_out['Github_Url'])
        print("github_content " , git_out)


    github_content = git_out or ""

    final_str = output + "\n" +  github_content + "\n"

    print(final_str)



    job_data , create = JobInsightData.objects.get_or_create(job=job)
    # print(job_data)
    # print(job_data.job_application_data)
    job_initial_data = job_data.job_application_data if job_data.job_application_data else ""
    print(job_initial_data)
    job_data.job_application_data  = job_initial_data + final_str
    job_data.save()

    return json_out
    



