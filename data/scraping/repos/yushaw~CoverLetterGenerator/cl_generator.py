# -*- coding: utf-8 -*-

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
import sys
import os


template = """
    You are an professional {profession}, and you are applying for a job at a company. You are writing a cover letter to apply for the job.
    You need to write a cover letter that is {length} words long. The cover letter should be written in a professional tone. 
    And you are provided with job description and your resume that you can use to write the cover letter, using markdown.
    
    the layout of the cover letter should be as follows:
    '''
    Employer or HR Manager’s Name

    Company Name
    Employer Street 
    Address City, State Zip code

    Dear Mr./Ms./Dr. Last Name of Addressee or Hiring Manager:

    Opening Paragraph
    State the position you are applying for, and how you found out about it, and ask for consideration based on the skills and experiences you have to offer. If you were referred by someone (ie: someone you know at the company, a recruiter you met at a career fair, etc) state that here. Make a general statement summarizing what qualifies you most for the job.
    2nd Paragraph
    In this section, you want to build a direct connection between the company’s needs and your background and skills. Stress what you have to offer, and avoid talking about what you want from them. Identify those parts of your experience that will interest THIS employer (refer to the job description if possible). You can draw attention to relevant coursework, special projects, and campus activities if they show a direct relationship to this position. Do not restate what’s in your resume, rather expand upon a specific project or accomplishment.
    3rd Paragraph – Optional
    Convince the employer that you have the personal qualities, passion, and motivation to succeed at this specific company. Relate your interests/passion to what you know about the company. (Convince the employer that you not only have the skills to do well at the job, but a vested interest in the company, industry, and the work itself.)
    Closing Paragraph
    Restate your interest in this position and how your unique qualifications fit the position. Request an interview, or tell the reader that you will contact him/her soon in order to schedule a mutually convenient time to meet. Thank the reader for his/her time and consideration.
    Sincerely,

    Your Signature

    Type Your Name
    '''

    job description: 
    '''
    {job_description}
    '''

    your resume:
    '''
    {resume}
    '''
    """
    
def generate_cover_letter(openai_api_key, 
                          template = template, 
                          my_resume = "", 
                          profession = "Product Manager", 
                          length = 300,
                          gpt_model = "gpt-4",
                          job_description = "",
                          jd_file_path = ""):
    cl_folder = 'cl'
    jd_abs_path = os.path.join(cl_folder, jd_file_path)
    
    llm = ChatOpenAI(model_name=gpt_model, openai_api_key=openai_api_key, temperature=0.9)
    prompt = PromptTemplate(template=template, input_variables=["profession", "length", "job_description", "resume"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    cl = llm_chain.run({ "profession": profession, "length": length, "job_description": job_description, "resume": my_resume})
       
    try:
        with open(jd_abs_path, 'w', encoding='utf-8') as file:
            file.write(cl)
        print(f"Successfully write to {jd_abs_path}")
    except Exception as e:
        print("Error encountered", str(e))