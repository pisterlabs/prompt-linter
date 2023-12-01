from langchain.tools import Tool, BaseTool
from pydantic import BaseModel, Field 
from typing import Any, Union, Tuple, Dict
from typing import Optional, Type

from langchain.agents.agent_toolkits.conversational_retrieval.tool import create_retriever_tool
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.messages import AIMessage


from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

from leadgen.utils.latex_utils import generate_latex, template_commands, render_latex
from leadgen.prompts.resume import generate_json_resume
from leadgen.llms.current import provider

from config import OPENAI_API_KEY
import random

RETRIEVAL_TEMPLATE = """
    Imagine you are a resume writer, hired by a company to figure out 
    the most relevant experiences for a candidate. Later on, your description will be condensed into information usng the following TypeScript Interface 
    for a JSON schema:

    interface Basics {
        name: string;
        email: string;
        phone: string;
        website: string;
        address: string;
    }

    Therefore, you want to be able to find the closest experiences to the job description below, and then
    use it to write a summary for each of the experiences you find, and do your best to include the items in the
    Basics schema. The job description is listed below:

    ### BEGIN JOB DESCRIPTION ### 

    <job_description>

    ### END JOB DESCRIPTION ### 
"""

#Source: https://github.com/vincanger/coverlettergpt/
COVER_LETTER_PROMPT = """
You are a cover letter generator. You will be given a job description, and a summary of the job applicant's experiences.
Below is the job description for the company.

### BEGIN JOB DESCRIPTION ### 

{job_description}

### END JOB DESCRIPTION ###


### BEGIN APPLICANT RESUME SUMMARY

{cv_summary}

### END APPLICANT RESUME SUMMARY

Write a cover letter for the applicant that matches their past experiences from the resume with the job description. Write the cover letter in the same language as the job description provided!
Rather than simply outlining the applicant's past experiences, you will give more detail and explain how those experiences will help the applicant succeed in the new job.

You will write the cover letter in a modern, professional style without being too formal, as a modern employee might do naturally.`,
"""


COVER_LETTER_WITH_A_WITTY_REMARK = """You are a cover letter generator.
You will be given a job description along with the job applicant's resume.

### BEGIN JOB DESCRIPTION ### 

{job_description}

### END JOB DESCRIPTION ###


### BEGIN APPLICANT RESUME SUMMARY

{cv_summary}

### END APPLICANT RESUME SUMMARY

Write a cover letter for the applicant that matches their past experiences from the resume with the job description. Write the cover letter in the same language as the job description provided!
Rather than simply outlining the applicant's past experiences, you will give more detail and explain how those experiences will help the applicant succeed in the new job.
You will write the cover letter in a modern, relaxed style, as a modern employee might do naturally.
Include a job related joke at the end of the cover letter.
"""

'''
Iteration 1 - Just takes experiences and creates a resume out of them
Iteration 2 - Actively searches for company information  
'''
COVER_LETTER_PROMPTS = [COVER_LETTER_PROMPT, COVER_LETTER_WITH_A_WITTY_REMARK]

class CreateCoverLetterInput(BaseModel):
    company_name: str = Field()
    job_description: str = Field()


class CreateCoverLetterTool(BaseTool):
    name = "create_cover_letter_from_experiences"

    #Later, we can add formatting, and graphics 
    description = """Use this tool to create a personalized cover letter.  

    Provide the company name and job description, and then this tool will

    1. Retrieve the k most similar experiences 

    2. Construct a cover letter and save it to a text file called cover.txt

    """

    args_schema: Type[BaseModel] = CreateCoverLetterInput

    def __init_subclass__(cls, **kwargs: Any) -> None:
        cls.userdb = kwargs['userdb']

    def _run(
        self, company_name: str, job_description: str, run_manager = None
    ) -> str:
        """Use the tool."""

        print('Cover letter', company_name, job_description)
        llm = self.userdb.get_llm()
        qa = self.userdb.get_qa_chain()
                
        #Generate a summary to be used for the cover letter
        prompt = RETRIEVAL_TEMPLATE.replace("<job_description>", job_description)
        cv_summary = qa(
            {"question": prompt, 
            "chat_history" : []}, 
            return_only_outputs=True
        )["answer"]

        print('summary', cv_summary)  

        prompt = COVER_LETTER_PROMPTS[random.randint(0, 100) % len(COVER_LETTER_PROMPTS)].format(job_description = job_description, cv_summary = cv_summary)
        cover_letter = llm.invoke(prompt)

        if type(cover_letter) == AIMessage:
            cover_letter = cover_letter.content

        print(cover_letter)
        with open('cover.txt', 'w') as f:
            f.write(cover_letter)

        return f'Cover letter saved as cover.txt!'

    async def _arun(
        self, company_name: str, job_description: str, run_manager = None
    ) -> str:
        """Use the tool."""

        print('Cover letter', company_name, job_description)
        llm = self.userdb.get_llm()
        qa = self.userdb.get_qa_chain()

        #Generate a summary to be used for the cover letter
        prompt = RETRIEVAL_TEMPLATE.replace("<job_description>", job_description)
        cv_summary = qa(
            {"question": prompt, 
            "chat_history" : []}, 
            return_only_outputs=True
        )["answer"]

        print('summary', cv_summary)  

        prompt = COVER_LETTER_PROMPTS[random.randint(0, 100) % len(COVER_LETTER_PROMPTS)].format(job_description = job_description, cv_summary = cv_summary)
        cover_letter = llm.invoke(prompt)

        if type(cover_letter) == AIMessage:
            cover_letter = cover_letter.content

        with open('cover.txt', 'w') as f:
            f.write(cover_letter)

        return f'Cover letter saved as cover.txt!'
