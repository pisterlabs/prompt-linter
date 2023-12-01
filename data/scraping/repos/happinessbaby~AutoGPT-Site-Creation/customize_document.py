from basic_utils import read_txt
from openai_api import get_completion
from openai_api import get_completion, get_completion_from_messages
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, Tool, AgentExecutor
from pathlib import Path
from basic_utils import read_txt, convert_to_txt
from langchain_utils import create_search_tools, generate_multifunction_response, create_summary_chain
from common_utils import get_generated_responses, get_web_resources, extract_posting_keywords, retrieve_from_db, get_web_resources, extract_pursuit_information
from typing import Any, List, Union, Dict
from langchain.docstore.document import Document
from langchain.tools import tool
import json
from json import JSONDecodeError
import faiss
import asyncio
import random
import base64
from datetime import date
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


def customize_personal_statement(personal_statement="", about_me="", program_path=""):

    personal_statement = read_txt(personal_statement)
    institution=""
    program = ""
    institution_description=""
    program_description=""
    program_specification=""
    if (Path(program_path).is_file()):
        posting = read_txt(program_path)
        prompt_template = """Identity the program, institution then provide a summary in 100 words or less of the following program:
            {text} \n
            Focus on the uniqueness, classes, requirements involved. Do not include information irrelevant to this specific program.
        """
        program_specification = create_summary_chain(program_path, prompt_template, chunk_size=4000)
        pursuit_dict = extract_pursuit_information(posting)
        program = pursuit_dict.get("program", "")
        institution = pursuit_dict.get("institution", "")
    elif about_me!="":
        pursuit_dict = extract_pursuit_information(about_me)
        program = pursuit_dict.get("program", "")
        institution = pursuit_dict.get("institution", "")
        if program!=-1:
            program_query = f"""Research the degree program in the institution provided below. 
            Find out what {program} at the institution {institution} involves, and what's special about the program, and why it's worth pursuing.    
            In 100 words or less, summarize your research result.  
            If institution is -1, research the general program itself.
            """
            program_description = get_web_resources(program_query)    
    if institution!=-1:
        institution_query = f""" Research {institution}'s culture, mission, and values.   
                        In 50 words or less, summarize your research result.                     
                        Look up the exact name of the institution. If it doesn't exist or the search result does not return an institution output -1."""
        institution_description = get_web_resources(institution_query)

    query = f""" You are to help Human polish a personal statement. 
    Prospective employers and universities may ask for a personal statement that details qualifications for a position or degree program.
    You are provided with a personal statement and various pieces of information, if available, that you may use.
    personal statement: {personal_statement}
    program specification: {program_specification}
    institution description: {institution_description}
    program description: {program_description}

    please consider blending these information into the existing personal statement. 
    Use the same tone and style when inserting information.
    Correct grammar and spelling mistakes. 

    """
    tools = create_search_tools("google", 3)
    response = generate_multifunction_response(query, tools)
    return response

def customize_cover_letter(cover_letter="", about_me="", posting_path="",):

    cover_letter = read_txt(cover_letter)
    company_description = ""
    job_specification = ""
    job_description = ""
    job = ""
    company = ""
    if (Path(posting_path).is_file()):
        posting = read_txt(posting_path)
        prompt_template = """Identity the job position, company then provide a summary in 100 words or less of the following job posting:
            {text} \n
            Focus on roles and skills involved for this job. Do not include information irrelevant to this specific position.
        """
        job_specification = create_summary_chain(posting_path, prompt_template, chunk_size=4000)
        pursuit_dict = extract_pursuit_information(posting)
        company = pursuit_dict.get("company", "")
        job = pursuit_dict.get("job", "")
    elif about_me!="":
        pursuit_dict = extract_pursuit_information(about_me)
        job = pursuit_dict.get("job", "")
        company = pursuit_dict.get("company", "")
        if job!="":
            job_query  = f"""Research what a {job} does and output a detailed description of the common skills, responsibilities, education, experience needed. 
                            In 100 words or less, summarize your research result. """
            job_description = get_web_resources(job_query)  
    if company  !="":
        company_query = f""" Research what kind of company {company} is, such as its culture, mission, and values.       
                            In 50 words or less, summarize your research result.                 
                            Look up the exact name of the company. If it doesn't exist or the search result does not return a company, output -1."""
        company_description = get_web_resources(company_query)
    
    query = f""" Your task is to help Human revise and polish a cover letter. You are provided with a cover letter and various pieces of information, if available, that you may use.
    Foremost important is the applicant's about me that you should always keep in mind.
    The applicant is applying for job {job} at company {company}. Please change the cover letter's existing information to suit the purpose of applying to this job and this company, if available.
    Please consider blending the below information into the existing cover letter to make it more descriptive, personalized, and appropriate for the occasion.

    about me: {about_me}
    cover letter: {cover_letter}
    job desciption: {job_description}
    job specification: {job_specification}
    company description: {company_description}

    Use the same tone and style when inserting information.
    Correct grammar and spelling mistakes. 

    """

    tools = create_search_tools("google", 3)
    response = generate_multifunction_response(query, tools, early_stopping=True)
    return response

def customize_resume(resume="", about_me="", posting_path=""):

    resume_content = read_txt(resume)
    # generated_info_dict=get_generated_responses(resume_content = resume_content, about_me=about_me, posting_path="")
    if (Path(posting_path).is_file()):
        posting = read_txt(posting_path)
        keywords = extract_posting_keywords(posting)
    elif about_me!="":
        pursuit_dict = extract_pursuit_information(about_me)
        job = pursuit_dict["job"]
        program = pursuit_dict["program"]
        if job!=-1:
            keywords = get_web_resources(f"Research some ATS-friendly keywords and key phrases for {job}.")
        elif program!=-1:
            #TODO
            keywords = get_web_resources(f"Research")

    # advice_query = f"""ATS-friendly way to write a resume"""
    # advice = retrieve_from_db(advice_query)
    query = f"""  You are an expert resume advisor. 

        Generate a list of relevant information that can be added to or replaced in the resume.
        
        The most important piece of information is the list of ATS keywords and key phrases that are in the job description, if available. 

        You should use it as a your primarily guideline. 

        resume content: {resume_content}\n

        job description keywords: {keywords} \n

        Please provide your reasoning as well. Please format your output as in the following example;

        Things to add or replace in the resume:

        1. Communication skills: communication skills is listed in the job description but not in the resume

        2. Extract, Transform, and Load: ETL in the resume should be changed to Extract, Transform, and Load as in the job description

        The above is just an example for your reference. Do not let it be your answer. 
        
        """
            
    tools = create_search_tools("google", 3)
    response = generate_multifunction_response(query, tools)
    return response




def create_resume_customize_writer_tool() -> List[Tool]:

    """ Agent tool that calls the function that customizes resume. """

    name = "resume_customize_writer"
    parameters = '{{ "job_post_file":"<job_post_file>", "about_me":"<about_me>", "resume_file":"<resume_file>"}}'
    description = f""" Customizes and tailors resume. 
    Input should be a single string strictly in the following JSON format: {parameters}
    (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else)"""
    tools = [
        Tool(
        name = name,
        func = process_resume,
        description = description, 
        verbose = False,
        handle_tool_error=True,
        )
    ]
    print("Succesfully created resume customize wrtier tool.")
    return tools

def create_cover_letter_customize_writer_tool() -> List[Tool]:

    """ Agent tool that calls the function that customizes cover letter. """

    name = "cover_letter_customize_writer"
    parameters = '{{"cover_letter_file":"<cover_letter_file>", "about_me":"<about_me>", "job_post_file:"<job_post_file>"}}'
    description = f""" Customizes, improves, and tailors cover letter. 
    Input should be a single string strictly in the following JSON format: {parameters}
    (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else)"""
    tools = [
        Tool(
        name = name,
        func = process_cover_letter,
        description = description, 
        verbose = False,
        handle_tool_error=True,
        )
    ]
    print("Succesfully created cover letter customize writer tool.")
    return tools

def create_personal_statement_customize_writer_tool() -> List[Tool]:

    """ Agent tool that calls the function that customizes personal statement """

    name = "personal_statement_customize_writer"
    parameters = '{{"personal_statement_file":"<personal_statement_file>", "about_me":"<about_me>", "education_program_file:"<education_program_file>"}}'
    description = f""" Customizes, tailors, and improves personal statement.
    Input should be a single string strictly in the following JSON format: {parameters}
    (remember to respond with a markdown code snippet of a JSON blob with a single action, and NOTHING else)"""
    tools = [
        Tool(
        name = name,
        func = process_personal_statement,
        description = description, 
        verbose = False,
        handle_tool_error=True,
        )
    ]
    print("Succesfully created personal statement customize writer tool.")
    return tools

def process_cover_letter(json_request:str) -> str:

    try:
        json_request = json_request.strip("'<>() ").replace(" ", "").__str__().replace("'", '"')
        args = json.loads(json_request)
    except JSONDecodeError as e:
        print(f"JSON DECODER ERROR: {e}")
        return "Format in JSON and try again."
    
    if ("cover_letter_file" not in args or args["cover_letter_file"]=="" or args["cover_letter_file"]=="<cover_letter_file>"):
        return """stop using or calling the cover_letter_customize_writer tool. Ask user to upload their cover letter instead. """
    else:
        cover_letter = args["cover_letter_file"]
    if ("about_me" not in args or args["about_me"] == "" or args["about_me"]=="<about_me>") and ("job_post_file" not in args or args["job_post_file"]=="" or args["job_post_file"]=="<job_post_file>"):
        return """stop using or calling the resume_customize_writer tool. ASk user to provide job positing or describe which position to tailor their cover letter to."""
    else:
        if ("about_me" not in args or args["about_me"] == "" or args["about_me"]=="<about_me>"):
            about_me = ""
        else:
            about_me = args["about_me"]
        if ("job_post_file" not in args or args["job_post_file"]=="" or args["job_post_file"]=="<job_post_file>"):
            posting_path = ""
        else:
            posting_path = args["job_post_file"]
    # if ("resume_file" not in args or args["resume_file"]=="" or args["resume_file"]=="<resume_file>"):
    #     resume = ""
    # else:
    #     resume = args["resume_file"]

    return customize_cover_letter(cover_letter=cover_letter, about_me=about_me, posting_path=posting_path)

def process_personal_statement(json_request:str) -> str:

    try:
        json_request = json_request.strip("'<>() ").replace(" ", "").__str__().replace("'", '"')
        args = json.loads(json_request)
    except JSONDecodeError as e:
        print(f"JSON DECODER ERROR: {e}")
        return "Format in JSON and try again."
    
    if ("personal_statement_file" not in args or args["personal_statement_file"]=="" or args["personal_statement_file"]=="<personal_statement_file>"):
        return """stop using or calling the personal_statement_customize_writer tool. Ask user to upload their personal statement instead. """
    else:
        personal_statement = args["personal_statement_file"]
    if ("about_me" not in args or args["about_me"] == "" or args["about_me"]=="<about_me>") and ("education_program_file" not in args or args["education_program_file"]=="" or args["education_program_file"]=="<education_program_file>"):
        return """stop using or calling the resume_customize_writer tool. ASk user to provide program information or describe which program to tailor their personal statement to."""
    else:
        if ("about_me" not in args or args["about_me"] == "" or args["about_me"]=="<about_me>"):
            about_me = ""
        else:
            about_me = args["about_me"]
        if ("education_program_file" not in args or args["education_program_file"]=="" or args["education_program_file"]=="<education_program_file>"):
            program_path = ""
        else:
            program_path = args["education_program_file"]
    # if ("resume_file" not in args or args["resume_file"]=="" or args["resume_file"]=="<resume_file>"):
    #     resume = ""
    # else:
    #     resume = args["resume_file"]

    return customize_personal_statement(personal_statement=personal_statement, about_me=about_me, program_path=program_path)




def process_resume(json_request: str) -> str:

    try:
        json_request = json_request.strip("'<>() ").replace(" ", "").__str__().replace("'", '"')
        args = json.loads(json_request)
    except JSONDecodeError as e:
        print(f"JSON DECODER ERROR: {e}")
        return "Format in JSON and try again."
    print(args)
    if ("resume_file" not in args or args["resume_file"]=="" or args["resume_file"]=="<resume_file>"):
        return """stop using or calling the resume_customize_writer tool. Ask user to upload their resume instead. """
    else:
        resume = args["resume_file"]
    if ("about_me" not in args or args["about_me"] == "" or args["about_me"]=="<about_me>") and ("job_post_file" not in args or args["job_post_file"]=="" or args["job_post_file"]=="<job_post_file>"):
        return """stop using or calling the resume_customize_writer tool. ASk user to provide job positing or program information or describe which position or program to tailor their resume to."""
    else:
        if ("about_me" not in args or args["about_me"] == "" or args["about_me"]=="<about_me>"):
            about_me = ""
        else:
            about_me = args["about_me"]
        if ("job_post_file" not in args or args["job_post_file"]=="" or args["job_post_file"]=="<job_post_file>"):
            posting_path = ""
        else:
            posting_path = args["job_post_file"]

    return customize_resume(resume=resume, about_me=about_me, posting_path=posting_path)


if __name__=="__main__":
    personal_statement = "./uploads/file/personal_statement.txt"
    resume = "./resume_samples/resume2023v3.txt"
    posting_path = "./uploads/link/software07.txt"
    # customize_personal_statement(about_me="i want to apply for a MSBA program at university of louisville", personal_statement = personal_statement)
    cover_letter = "./generated_responses/cv01.txt"
    customize_cover_letter(about_me = "", cover_letter=cover_letter, posting_path=posting_path)
    # customize_resume (about_me="", resume=resume, posting_path=posting_path)
        