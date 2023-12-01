import openai
import json
import os
import sys
from PyPDF2 import PdfReader
import webbrowser

def open_web_page(url):
    webbrowser.open(url)

def clear_screen():
    _ = os.system('clear')

def validate_json(response):
    json_valid = is_json(response)
    if not json_valid:
        print("ChatGPT Error: Please try again.")
        print(response)
        sys.exit()
    try:
        data = json.loads(response) 
    except Exception as e:
        print(e)
        sys.exit()

    return data

def is_json(json_string):
    try:
        json_data = json.loads(json_string)
        return True
    except ValueError as e:
        print("Invalid JSON:", e)
        return False

def add_prompt_messages(role, content, messages):
    json_message = {
        "role": role, 
        "content": content
    }
    messages.append(json_message)
    return messages

def parse_file_name(file_name: str):
    name, extension = os.path.splitext(file_name)
    extension = extension.lstrip('.')
    return name, extension


def get_chat_completion_messages(messages, model="gpt-3.5-turbo", temperature=0.0): 
    try:
        response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    except Exception as e:
        print(e)
        sys.exit()
    else:
        return response.choices[0].message["content"]


def create_prompt_job_interview_questions(number_of_questions):
    bevavioral = int(number_of_questions * .20)
    technical = int(number_of_questions * .20)
    situational = int(number_of_questions * .25)
    competency = int(number_of_questions * .20)
    open_ended = int(number_of_questions * .05)
    closed_ended = int(number_of_questions * .05)
    personal = int(number_of_questions * .05)


    prompt = f"""
        Your task is to generate a list of {number_of_questions} common job interview questions
        
        It is important that the quesitons should be relevant to the job posting, and the resume of the candidate.

        Distribute your questions using the following categories and percentages of how to distribute the questions:
        
        - Behavioral = {bevavioral} should be behavioral questions.
        - Technical = {technical} should be technical questions.
        - Situational = {situational} should be situational questions.
        - Competency = {competency} should be competency questions.
        - Open-ended = {open_ended} should be open-ended questions.
        - Closed-ended = {closed_ended} should be closed-ended questions.
        - Personal = {personal} should be personal questions.
        
        Very important to only respond in JSON format, with the following keys:
        category:
        question:

        double check your response is in JSON format.
        """
    return prompt


def create_prompt_to_improve_resume(number_of_items):
    prompt = f"""
        i really want to improve my resume You better align with the job description and improve my score, create me a list of {number_of_items} things to improve my resume.
        Very important to only respond in JSON format, with the following keys:
        id:
        improvement:

        double check your response is in JSON format.
        """
    return prompt


def create_prompt_pro_con(number_of_items):
    prompt = f"""
        You are a hiring manager. Your task is to generate a list of {number_of_items} pros and cons for a job candidate.

        Very important to only respond in JSON format, with the following keys:
        pros:
        cons:

        double check your response is in JSON format.
        """
    return prompt


def create_system_prompt():
    system_prompt = f"""
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

    As a hiring manager, your responsibility is to evaluate the qualifications and skills of candidates for a specific job opening. Your task is to generate a relevance score that indicates how closely the candidate's resume matches the job description. The scoring scale ranges from 0 to 100, with 100 representing a perfect match and 0 indicating no match.

    To calculate the relevance score, compare the job description with the candidate's resume and assess the alignment of their qualifications and skills. Consider factors such as relevant work experience, education, certifications, and any specific requirements mentioned in the job description.

    Also, include your reasoning for the score.

    double check your score, reasoning and resume facts before submitting it to the hiring manager.
    """
    return system_prompt


def create_prompt_job_match_v2(job_description, resume):
    prompt = f"""
Here's a template you can use to generate the relevance score:
Here's are some examples way to do it:

1. Identify the key requirements in the job description.
2. Compare these requirements with the candidate's resume.
3. Assign points based on the alignment.
4. Add up the points to get the total score. In this case
5. Make sure to point out missing requirements in education, job duties or any others

Very important to only respond in JSON format, with the following keys:
    reasoning:
    score:
    missing_requirements:

    job_description = ```{job_description}```

    resume = ```{resume}```

    double check your response is in JSON format.
    """
    return prompt


def create_system_prompt_html():
    system_prompt = f"""
    You are a html web designer with outstanding design skills and a passion for creating websites.
    """
    return system_prompt


def create_prompt_html_report(data):
    prompt = f"""
    Your task is to take the data and create a fantasic HTML page.
    Please use colors to highlight important text.
    colors: 
    blue: #000080 for titles
    red: #A52A2A for missing requirements, cons, or negative feedback from reasoning or score below minimum_score
    green: #228B22 for pros, or positive feedback from reasoning, or score equal to or above minimum_score

    You will receive the data in the format of a JSON object.

    Group the questions by category here is an example: 
    category: Behavioral
    - Question

    Here is your json data enclosed and three backticks: ```{data}```

    you must output your results in HTML format and only the report you have created and nothing more.
    """
    return prompt


def get_resume_and_job_description(job_description_file_name, resume_file_name):

    resume_name, resume_format = parse_file_name(resume_file_name)

    # import job description, convert to text and save to file
    print("***** Importing job description: " + job_description_file_name)
    job_description = import_text_file("job_description/" + job_description_file_name)
    if job_description:
        print("+++++ Job description was imported.")
    else:
        print("----- No job description was imported.")

    print("***** Importing resume: " + resume_file_name)
    if resume_format == "pdf":
        resume = convert_pdf_to_text("resume/" + resume_file_name)
        if resume:
            print("+++++ Resume was imported.")
        else:
            print("----- No PDF resume was imported.")
    elif resume_format == "txt":
        # import resume, convert to text and save to file
        resume = import_text_file("resume/" + resume_file_name)
        if resume:
            print("+++++ Resume was imported.")
        else:
            print("----- No TXT resume was imported.")
    else:  
        print("----- Resume format is not supported. Exiting.")

    if not job_description or not resume:
        print("!!!!! No job description or resume was imported. Exiting.")
        sys.exit()

    return job_description, resume

# import a pdf and convert to text
def convert_pdf_to_text(path):
    try:
        with open(path, "rb") as filehandle:
            pdf = PdfReader(filehandle)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except FileNotFoundError:
        print(f"The file {path} does not exist.")
    except:
        print("An unexpected error occurred.")

# import a text file
def import_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except IOError:
        print(f"Error: Unable to read file '{file_path}'.")

# write text to a file
def write_text_to_file(filename, text):
    try:
        with open(filename, 'w') as f:
            f.write(text)
        print(f"Text successfully written to {filename}")
    except:
        print("An error occurred while writing the file.")
