import os 
import uuid

import json 
from langchain.vectorstores.faiss import FAISS


from langchain.tools import Tool
from langchain.document_loaders.json_loader import JSONLoader
from pydantic.v1 import BaseModel
from pydantic.v1 import Field
from typing import Union, Tuple, Dict, List
from typing import Optional, Type

from leadgen.llms.base import BaseLLM
from .adapters.MockAdapter import MockAdapter

class JobsDatabase:
    EMBED_SAVE_DIR = "embeddings_job"
    EMBED_SAVE_INDEX = "job_embeddings"
    JOBS_SAVE_DIR = "job_documents"

    def __init__(self, provider: BaseLLM, persist_directory = os.path.join("data", "jobs"), ) -> None:

        #Store our provider
        self.provider = provider

        #Load our vectorstore
        self.persist_dir = str(persist_directory)

        self.embeddings_dir = os.path.join(persist_directory, self.EMBED_SAVE_DIR)
        self.jobs_dir       = os.path.join(persist_directory, self.JOBS_SAVE_DIR)

        print(self.embeddings_dir, self.jobs_dir)

        #Create the directory if it doesn't exist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.jobs_dir, exist_ok=True)

        _, self.vectorstore = self.load_vectorstore()

        #Setup our adapters 
        self.adapters = {'MockAdapter' : MockAdapter()} #name --> adapter, i.e adapters["linkedin"] contains a linkedin adapter


    def load_vectorstore(self):
        if os.path.exists(os.path.join(self.embeddings_dir, f'{self.EMBED_SAVE_INDEX}.faiss')):
            return True, FAISS.load_local(self.embeddings_dir, index_name=self.EMBED_SAVE_INDEX, embeddings=self.provider.get_embeddings())

        return False, FAISS.from_texts(["Dummy text."], self.provider.get_embeddings()) 

    def save_vectorstore(self):
        self.vectorstore.save_local(self.embeddings_dir, index_name=self.EMBED_SAVE_INDEX)

    def store_job(self, job):

        #Store the job in the local directory
        print("STORING JOB")

        job_name = uuid.uuid4()
        fp = os.path.join(self.jobs_dir, f'{job_name}.json')
        with open(fp, 'w') as f:
            json.dump(job, f)


        #Adding the documents
        doc_loader = JSONLoader(file_path=fp, jq_schema='.', text_content=False)
        docs = doc_loader.load()

        self.vectorstore.add_documents(docs)

        print("STORED JOB", job_name)
        return str(job_name)
        
    def store_jobs(self, jobs):
        uuids = []
        for job in jobs:
            uuids.append(self.store_job(job))

        self.save_vectorstore()
        return uuids

    def get_retriever(self, k = 5):
        return self.vectorstore.as_retriever(k = k)

    def get_job(self, providedUUID):

        print('getting job', providedUUID)

        fp = os.path.join(self.jobs_dir, f'{providedUUID}.json')
        if not os.path.exists(fp):
            return f"Use the  tool to return a job first! Re-run the procedure, and call obtain_job_data"

        with open(os.path.join(self.jobs_dir, f'{providedUUID}.json'), 'r') as f:
            partial_job = json.load(f)
            print('sending', providedUUID)
            return partial_job
    
    def create_tool_get_job(self):
        class ObtainDataByUUID(BaseModel):
            uuid: str = Field()

        def obtain_data_by_uuid(uuid):
            print("getting job by uuid")
            return self.get_job(uuid)

        description = """Retrieve some jobs using the obtain_job_data tool, and then ask about the job using it's uuid.

        Make sure to run this tool only after calling the obtain_job_data tool 
        """

        tool = Tool.from_function(
            func=obtain_data_by_uuid,  
            name="job_retrieval_search",
            description=description, 
            args_schema=ObtainDataByUUID
        )
 
        return tool

    def retrieve_jobs(self, topic, k = 1):
        '''
            Returns jobs based on a simple keyword search 
        '''

        jobs = []
        for key in self.adapters:
            jobs.extend(self.adapters[key].retrieve_jobs(topic))
            
            if len(jobs) > k:
                jobs = jobs[:k]
                break 

        uuids = self.store_jobs(jobs)
        return f'Successfully retrieved {len(jobs)} on {topic}! Use the get_jobs tool to obtain data on each job using the uuids listed here: {",".join(uuids)} by passing individual ones into the get_jobs tool as input.'
    
    def create_tool_retrieve_jobs(self):

        class ObtainJobDataSchema(BaseModel):
            get_data_topic: str = Field()

        def get_job_data(topic):
            print("retrieving jobs", topic)
            return self.retrieve_jobs(topic)

        description = """Gets and obtains job data. Use this tool BEFORE any data analysis on job postings companies have made, and the requirements they are looking for within a job. Without running this tool first, you won't
        Example input: data analyst

        This would get you job postings from companies looking for data analysts. You can ALSO use this tool in sucession if you want data on multiple topics. For example, you might realize that
        after getting data on data analytics, some machine learning jobs might also be relevant. Then re-run this tool, and it'll add machine learning jobs as well

        Additionally, you will have to specify the number of jobs you'll need. If no clear wording is given,
        default to 5.
        """

        tool = Tool.from_function(
            func=get_job_data,  
            name="obtain_job_data",
            description=description, 
            args_schema=ObtainJobDataSchema
        )
 
        return tool

    def get_complete_application(self, uuid):
        print('get complete application', uuid)
        with open(os.path.join(self.jobs_dir, f'{uuid}.json'), 'r') as f:
            partial_job = json.load(f)
            print('get complete application done', partial_job)
            return f'Here are the application questions you\'ll need to answer for application {uuid}. Answer them, and then submit the job application with the resume and cover letter. Questions: {self.adapters[partial_job["metadata"]["src"]].get_complete_application(partial_job)}'
            
    def create_tool_get_application(self):

        class GetApplicationQuestionsSchema(BaseModel):
            uuid: str = Field()

        def get_app_qs(uuid):
            return self.get_complete_application(uuid)
        
        print(isinstance(GetApplicationQuestionsSchema, BaseModel))

        tool = Tool.from_function(
            func=get_app_qs,  
            name="get_job_application_questions",
            description="Retrieve all the questions you need to fill out for the job application",
            args_schema=GetApplicationQuestionsSchema
        )
 
        return tool

    def apply_to_application(self, uuid, answers, resume_fp, content_letter_fp):
        print('apply to applications', uuid)
        with open(os.path.join(self.jobs_dir, f'{uuid}.json'), 'r') as f:
            partial_job = json.load(f)
            print('apply to applications done', uuid)
            return self.adapters[partial_job["metadata"]["src"]].apply_to_application(partial_job, answers, resume_fp, content_letter_fp)

    def create_tool_apply_job(self):

        class JobAnswer(BaseModel):
            content: str

        class JobAnswers(BaseModel):
            answers: List[JobAnswer]

        class ApplyToJobSchema(BaseModel):
            uuid: str = Field()
            answers: JobAnswers = Field()
            resume_file_filepath: str = Field()
            content_letter_file_filepath: str = Field()

        def apply_to_app(uuid, answers, resume_fp, content_letter_fp):
            return self.apply_to_application(uuid, json.load(answers), resume_fp, content_letter_fp)

        tool = Tool.from_function(
            func=apply_to_app,  
            name="apply_to_application",
            description="Use this tool to apply to a job",
            args_schema=ApplyToJobSchema
        )
 
        return tool

    def poll_job_application_status(self, uuid):
        with open(os.path.join(self.jobs_dir, f'{uuid}.json'), 'r') as f:
            partial_job = json.load(f)
            return self.adapters[partial_job["metadata"]["src"]].poll_job_application_status(partial_job)

    def create_tool_application_status(self):

        class ApplicationStatusSchema(BaseModel):
            uuid: str = Field()

        def poll_job_app_status(uuid):
            return self.poll_job_application_status(uuid)

        tool = Tool.from_function(
            func= poll_job_app_status, 
            name="poll_job_application_status",
            description="Use this tool to check in on application status if ever requested by the user.",
            args_schema=ApplicationStatusSchema
        )
 
        return tool

    def get_toolkit(self):
        return [
            self.create_tool_get_application(),
            self.create_tool_apply_job(),
            self.create_tool_application_status(),
            self.create_tool_retrieve_jobs(),
            self.create_tool_get_job(),
        ]
    
    