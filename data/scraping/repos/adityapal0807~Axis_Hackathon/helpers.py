import docx
import PyPDF2
import os
import openai
import requests
import os
from dotenv import load_dotenv
import json
import ast
from io import BytesIO
from django.conf import settings
from django.core.mail import send_mail
import uuid

from .prompts import sample_job_description,initial_prompt,resume_ranker,question_generator,role_resume_score,role_resume_ranker,resume_prompt_data,role_jd_suggestor,jd_prompt_creator,role_question_generator,question_generator,role_situation,create_situation,analyse_situational_answer

load_dotenv()

def convert_to_text(file_data, file_extension):
    if file_extension == '.docx':
        return convert_docx_to_text(file_data)
    elif file_extension == '.pdf':
        return convert_pdf_to_text(file_data)
    else:
        return None

def convert_docx_to_text(docx_file):
    try:
        doc = docx.Document(BytesIO(docx_file.read()))
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error converting DOCX file: {e}")
        return None
    
def convert_pdf_to_text(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        # Extract text from each page of the PDF
        extracted_text = ''
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()
        return extracted_text
    except Exception as e:
        print(f"Error converting PDF file: {e}")
        return None
    

def role_classifier(system_role,user_role):
    message = [{
        "role": "system",
        "content": f"""{system_role}"""
        },
    {
        "role": "user",
        "content": f"""{user_role}"""
    }]
    return message


def jd_suggestor(job_description,model_name='gpt-3.5-turbo',temperature=0.7):

    #setting up the roles for model
    system_role = role_jd_suggestor
    user_role = jd_prompt_creator(job_description=job_description)

    roles = role_classifier(system_role=system_role,user_role=user_role)

    response = requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {os.getenv(f"OPENAI_API_KEY")}',
                        'Content-Type': 'application/json',
                    },
                    json={
                        'model': model_name,
                        'temperature': temperature,
                        'messages': roles,
                    },
                )
    response_data = json.loads(response.text)
    try:
        
        # Extract the 'content' field from the 'message' dictionary
        content = response_data['choices'][0]['message']['content']
        content = str(content).replace('\n','')
        content = ast.literal_eval(content)
        return content
    except Exception as e:
        print(f"Model Error: {e}")
        return {'Error':response_data}

def resume_scorer(job_description,resume_text,model_name='gpt-4',temperature=0.6):

    #setting up the roles for model
    system_role = role_resume_score
    user_role = initial_prompt(job_description=job_description,resume_text=resume_text)

    roles = role_classifier(system_role=system_role,user_role=user_role)

    response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': model_name,
                    'temperature': temperature,
                    'messages': roles,
                },
            )
    response_data = json.loads(response.text)
    try:
        # Extract the 'content' field from the 'message' dictionary
        content = response_data['choices'][0]['message']['content']
        content = json.loads(content)
        return content

    except Exception as e:
        print(f"Model Error: {e}")
        return {'Error':response_data}
    
def rank_resume(job_description,combined_resume_json,model_name='gpt-4',temperature=0.6):

    #setting up the roles for model
    system_role = role_resume_ranker
    user_role = resume_ranker(job_description=job_description,resumes_json_data=combined_resume_json)

    roles = role_classifier(system_role=system_role,user_role=user_role)

    response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': model_name,
                    'temperature': temperature,
                    'messages': roles,
                },
            )
    response_data = json.loads(response.text)
    try:

        # Extract the 'content' field from the 'message' dictionary
        content = response_data['choices'][0]['message']['content']
        content = str(content).replace('\n','')
        content = json.loads(content)
        return content

    except Exception as e:
        print(f"Model Error: {e}")
        return {'Error':response_data}
    
def generate_questions(job_description,model_name='gpt-4',temperature=0.6):
    #setting up the roles for model
    system_role = role_question_generator
    user_role = question_generator(job_description=job_description)

    roles = role_classifier(system_role=system_role,user_role=user_role)

    response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': model_name,
                    'temperature': temperature,
                    'messages': roles,
                },
                stream=False
            )
    response_data = json.loads(response.text)
    try:

        # Extract the 'content' field from the 'message' dictionary
        content = response_data['choices'][0]['message']['content']
        content = str(content).replace('\n','')
        content = json.loads(content)
        return content

    except Exception as e:
        print(f"Model Error: {e}")
        return {'Error':response_data}
    
def candidate_login_credential(email,password):
    subject = 'Test Credentials - Do Not Share!!'
    message = f""" 
        Hi {email},

        These are your credentials for the test.

        Login Email : {email}
        Password : {password}

        If you did not make this request then please ignore this email.

        Start your Test Here:
        http://127.0.0.1:8000/candidate

        """
    email_from = settings.EMAIL_HOST_USER
    recipient_list = [email]
    send_mail(subject,message,email_from,recipient_list)
    return True


def generate_situation(job_description,model_name='gpt-3.5-turbo',temperature=0.5):
    #setting up the roles for model
    system_role = role_situation
    user_role = create_situation(job_description=job_description)

    roles = role_classifier(system_role=system_role,user_role=user_role)

    response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': model_name,
                    'temperature': temperature,
                    'messages': roles,
                },
                stream=False
            )
    response_data = json.loads(response.text)
    try:

        # Extract the 'content' field from the 'message' dictionary
        content = response_data['choices'][0]['message']['content']
        content = str(content).replace('\n','')
        content = json.loads(content)
        return content

    except Exception as e:
        print(f"Model Error: {e}")
        return {'Error':response_data}

    
    
    
def candidate_situation_answer(situation,expected_answer,candidate_answer,model_name='gpt-4',temperature=0.7):
    #setting up the roles for model
    system_role = role_situation
    user_role =analyse_situational_answer(situation=situation,expected_answer=expected_answer,candidate_answer=candidate_answer)

    roles = role_classifier(system_role=system_role,user_role=user_role)

    response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                    'Content-Type': 'application/json',
                },
                json={
                    'model': model_name,
                    'temperature': temperature,
                    'messages': roles,
                },
                stream=False
            )
    response_data = json.loads(response.text)
    try:

        # Extract the 'content' field from the 'message' dictionary
        content = response_data['choices'][0]['message']['content']
        content = str(content).replace('\n','')
        content = json.loads(content)
        return content

    except Exception as e:
        print(f"Model Error: {e}")
        return {'Error':response_data}

    
    
    










