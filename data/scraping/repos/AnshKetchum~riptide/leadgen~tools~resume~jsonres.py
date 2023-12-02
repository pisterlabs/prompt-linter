
from langchain.tools import Tool, BaseTool
from pydantic import BaseModel, Field 
from typing import Union, Tuple, Dict
from typing import Optional, Type

from langchain.agents.agent_toolkits.conversational_retrieval.tool import create_retriever_tool
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

from leadgen.utils.latex_utils import generate_latex, template_commands, render_latex
from leadgen.llms.current import provider

from .prompts import BASICS_PROMPT, EDUCATION_PROMPT, AWARDS_PROMPT, PROJECTS_PROMPT, WORK_PROMPT, SKILLS_PROMPT, SYSTEM_TAILORING 
import json 
import random
from stqdm import stqdm

'''
Iteration 1 - Just takes experiences and creates a resume out of them
Iteration 2 - Actively searches for company information  
'''

class CreateResumeToolInput(BaseModel):
    company_name: str = Field()
    job_description: str = Field()


class CreateResumeTool(BaseTool):
    name = "create_resume_from_experiences"

    description = """Use this tool to create a personalized resume.  

    Provide the company name and job description, and then this tool will

    1. Retrieve the k most similar experiences and create a JSON resume

    2. Construct a LaTeX resume from that JSON resume, and save it as an output pdf under the filename 'output.pdf'.

    """

    args_schema: Type[BaseModel] = CreateResumeToolInput

    def _run(
        self, company_name: str, job_description: str, run_manager = None
    ) -> str:
        """Use the tool."""
        print('Generating Resume.')

        vectordb = FAISS.load_local('data', index_name="user_docs", embeddings= provider.get_embeddings())

        llm = provider.get_llm()
        qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectordb.as_retriever())
                
        #Generate a summary to be used as for CV
        sections = []
        for p in stqdm(
            [   BASICS_PROMPT, 
                EDUCATION_PROMPT,
                AWARDS_PROMPT,
                PROJECTS_PROMPT, 
                WORK_PROMPT,
                SKILLS_PROMPT, 
            ],
            desc="This might take a while..."
        ):
            prompt = p.replace("<job_description>", job_description)

            answer = qa(
                {"question": prompt, 
                "chat_history" : []}, 
                return_only_outputs=True
            )["answer"]

            answer = json.loads(answer)
        
            if prompt == BASICS_PROMPT and "basics" not in answer:
                answer = {"basics": answer}  # common mistake GPT makes

            sections.append(answer)

        json_resume = {}
        for section in sections:
            json_resume.update(section)

        print("JSON RESUME")
        print(json_resume)

        with open('json_resume.json', 'w') as f: 
            json.dump(json_resume, f)

        rand_choice = list(template_commands.keys())[random.randint(1, 100) % len(template_commands)]
        latex_resume = generate_latex(rand_choice, json_resume, ["education", "work", "skills", "projects", "awards"])
        resume_bytes = render_latex(template_commands[rand_choice], latex_resume)

        print('writing bytes')
        with open('out.pdf', 'wb') as f:
            f.write(resume_bytes)
        print('written bytes')
        print('done')
        return f'Resume saved as out.pdf!'

    async def _arun(
        self, company_name: str, job_description: str, run_manager = None
    ) -> str:
        """Use the tool."""

        print('Generating Resume.')

        vectordb = FAISS.load_local('data', index_name="user_docs", embeddings= provider.get_embeddings())

        llm = provider.get_llm()
        qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectordb.as_retriever())
                
        #Generate a summary to be used as for CV
        sections = []
        for p in stqdm(
            [   BASICS_PROMPT, 
                EDUCATION_PROMPT,
                AWARDS_PROMPT,
                PROJECTS_PROMPT, 
                WORK_PROMPT,
                SKILLS_PROMPT, 
            ],
            desc="This might take a while..."
        ):
            prompt = p.replace("<job_description>", job_description)

            answer = qa(
                {"question": prompt, 
                "chat_history" : []}, 
                return_only_outputs=True
            )["answer"]

            answer = json.loads(answer)
        
            if prompt == BASICS_PROMPT and "basics" not in answer:
                answer = {"basics": answer}  # common mistake GPT makes

            sections.append(answer)

        json_resume = {}
        for section in sections:
            json_resume.update(section)

        print("JSON RESUME")
        print(json_resume)

        with open('json_resume.json', 'w') as f: 
            json.dump(json_resume, f)

        rand_choice = list(template_commands.keys())[random.randint(1, 100) % len(template_commands)]
        latex_resume = generate_latex(rand_choice, json_resume, ["education", "work", "skills", "projects", "awards"])
        resume_bytes = render_latex(template_commands[rand_choice], latex_resume)

        with open('out.pdf', 'wb') as f:
            f.write(resume_bytes)

        return f'Resume saved as out.pdf!'
