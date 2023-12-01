from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from .parsers import ResumeParser, JobDescriptionParser

from threading import Thread

class Screener:
    def __init__(self, path: str, job_description: str):
        print("running constructor...")
        load_dotenv()

        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.path = path
        
        # Create threads for parser initializations
        def init_resume_parser():
            self.resume = ResumeParser(path)

        def init_job_description_parser():
            self.job_description = JobDescriptionParser(job_description)
        
        # Start the threads
        resume_thread = Thread(target=init_resume_parser)
        job_description_thread = Thread(target=init_job_description_parser)
        resume_thread.start()
        job_description_thread.start()
        
        # Wait for both threads to complete
        resume_thread.join()
        job_description_thread.join()
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def is_correct_fit(self):
        print ("running is_correct_fit...")
        prompt = ChatPromptTemplate.from_template("""
        Given the below job requirements, essential and nonessential, and the candidates resume, provide a YES or NO answer about whether the candidate is a truly good fit for the job.
        Job Description: {job_description}
        Resume: {resume}
        Answer:                                           
        """)

        rag_chain = (
            {"job_description": RunnablePassthrough(), "resume": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.invoke({"job_description": self.job_description.requirements, "resume": self.resume.resume})

    def strengths(self):
        print("running strengths...")

        prompt = ChatPromptTemplate.from_template("""
        Given the below job description and the candidates relevant skills and experiences, point out strengths about the candidate's experiences relative to the job, if any.
        If there are any strengths, respond in the following format (delimited by triple backticks) for each strength, one after another in the style of a python list:
        \"\"\"
        [pair1, pair2, ... pairN] where pairI =                                          
        (
        "the relevant snippet, verbatim, from the candidate's resume",
        "your commentary on the strength and why it's good for the job"                                                                              
        )
        \"\"\"                                                                                    
        Job Description: {job_description}
        Relevant Skills and Experiences: {context}
        Answer:                                           
        """)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)   

        loader = UnstructuredPDFLoader(self.path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(data)

        embedding = OpenAIEmbeddings(model='text-embedding-ada-002')

        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
        retriever = vectorstore.as_retriever()

        rag_chain = (
            {"job_description": RunnablePassthrough(), "context": retriever | format_docs}
            | prompt
            | self.llm
            | StrOutputParser()
        ) 

        return rag_chain.invoke(self.job_description.requirements)

    def weaknesses(self):
        print ("running weaknesses...")
        prompt = ChatPromptTemplate.from_template("""
        Given the below job requirements, if there are any requirements the candidate does not meet, definitively give your answers in the following format (delimited by triple backticks), one after another in the style of a python list:
        \"\"\"
        [pair1, pair2, ... pairN] where pairI =                                          
        (
        "the relevant snippet, verbatim, from the job requirements",
        "your commentary on what is lacking from the candidate's resume"                                                                              
        )
        \"\"\"        
        Job Requirements: {job_requirements}
        Resume: {resume}
        Answer:                                           
        """)

        rag_chain = (
            {"job_requirements": RunnablePassthrough(), "resume": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.invoke({"job_requirements": self.job_description.essentials, "resume": self.resume.resume})

    def suitable_jobs(self):
        print ("running suitable_jobs...")
        prompt = ChatPromptTemplate.from_template("""
        Given the below list of jobs, and the candidates relevant skills and experiences, pick 3 listed jobs that best suit the candidate.
        If there are any jobs that are suitable, respond in the following format (delimited by triple backticks) for each job, one after another in the style of a python list:
        \"\"\"
        (
        "the job title",
        )
        \"\"\"
        Job List:
        Accounting 
        Accounting and Finance 
        Account Management 
        Account Management/Customer Success 
        Administration and Office 
        Advertising and Marketing 
        Animal Care 
        Arts 
        Business Operations 
        Cleaning and Facilities 
        Computer and IT 
        Construction 
        Corporate 
        Customer Service 
        Data and Analytics 
        Data Science 
        Design 
        Design and UX Editor 
        Education 
        Energy Generation and Mining 
        Entertainment and Travel Services 
        Farming and Outdoors 
        Food and Hospitality Services 
        Healthcare 
        HR 
        Human Resources and Recruitment 
        Installation, Maintenance, and Repairs 
        IT 
        Law 
        Legal Services 
        Management 
        Manufacturing and Warehouse 
        Marketing 
        Mechanic 
        Media, PR, and Communications 
        Mental Health 
        Nurses 
        Office Administration 
        Personal Care and Services 
        Physical Assistant 
        Product 
        Product Management 
        Project Management 
        Protective Services 
        Public Relations 
        Real Estate 
        Recruiting 
        Retail 
        Sales 
        Science and Engineering 
        Social Services 
        Software Engineer 
        Software Engineering 
        Sports, Fitness, and Recreation 
        Transportation and Logistics 
        Unknown 
        UX 
        Videography 
        Writer 
        Writing and Editing
        Resume: {resume}
        Answer:                                           
        """)

        rag_chain = (
            {"resume": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.invoke({"resume": self.resume.resume})