import re
import docx2txt
import spacy
import fitz
import os
import replicate
import openai
import json
import streamlit as st


def get_category(data, category):
    try:
        value = data[category]
        if not isinstance(value, list):
            value = [value, ]
        return value
    except:
        return None


def extract_email(text):
    email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    if email:
        try:
            return [email[0].split()[0].strip(';'), ]
        except IndexError:
            return None


def extract_mobile_number(text):
    mob_num_regex = r'''(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)
                        [-\.\s]*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'''
    phone = re.findall(re.compile(mob_num_regex), text)
    if phone:
        number = ''.join(phone[0])
        return [number, ]


def get_text_from_docx(doc_path):
    try:
        temp = docx2txt.process(doc_path)
        text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
        return ' '.join(text)
    except KeyError:
        return ' '


def get_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = " "
    for page in doc:
        text = text + str(page.get_text())

    # file = open('resume_text.txt', 'w')
    # file.write(text)
    # file.close()

    return text


def extract_data(text, model="./Resources/Models/output/model-best"):
    nlp = spacy.load(model)
    doc = nlp(text)

    resume_data = {}
    data_str = ''
    for ent in doc.ents:
        if ent.label_ in resume_data.keys():
            resume_data[ent.label_].append(ent.text)
        else:
            resume_data[ent.label_] = [ent.text, ]
        data_str += ent.text + " -> " + ent.label_ + '\n'

    data_dict = {}

    data_dict["name"] = get_category(resume_data, "NAME")
    data_dict["phone number"] = extract_mobile_number(text)
    data_dict["email"] = extract_email(text)
    data_dict["linkedin"] = get_category(resume_data, "LINKEDIN LINK")
    data_dict["degree"] = get_category(resume_data, "DEGREE")
    data_dict["year of graduation"] = get_category(resume_data, "YEAR OF GRADUATION")
    data_dict["university"] = get_category(resume_data, "UNIVERSITY")
    data_dict["skills"] = get_category(resume_data, "SKILLS")
    data_dict["certification"] = get_category(resume_data, "CERTIFICATION")
    data_dict["awards"] = get_category(resume_data, "AWARDS")
    data_dict["worked as"] = get_category(resume_data, "WORKED AS")
    data_dict["companies worked at"] = get_category(resume_data, "COMPANIES WORKED AT")
    data_dict["years of experience"] = get_category(resume_data, "YEARS OF EXPERIENCE")
    data_dict["language"] = get_category(resume_data, "LANGUAGE")

    # print(data_dict)
    # file = open('resume_data.txt', 'w')
    # file.write(data_str)
    # file.close()

    return data_dict, data_str


def extract_data_llama(text):
    os.environ["REPLICATE_API_TOKEN"] = st.secrets.tokens.replicate
    prompt = generate_prompt(text)
    response = get_response_llama(prompt)
    try:
        clean_response = response[response.index('{'):]
        json_object = json.loads(clean_response)
        return json_object
    except ValueError:
        return response


def init_openai():
    openai.api_key = st.secrets.tokens.openai


def extract_data_chatgpt(text):
    prompt = generate_prompt(text)
    response = get_response_chatgpt(prompt)
    try:
        json_object = json.loads(response)
        return response, json_object
    except ValueError:
        return response, None


def generate_questions_chatgpt(data):
    prompt = generate_questions_prompt(data)
    response = get_response_chatgpt(prompt)
    try:
        json_object = json.loads(response)
        return response, json_object
    except ValueError:
        return response, None


def analyze_resume_chatgpt(text, job_desc):
    prompt = analyze_resume_prompt(text, job_desc)
    response = get_response_chatgpt(prompt)
    try:
        json_object = json.loads(response)
        return response, json_object
    except ValueError:
        return response, None


def generate_prompt(text):
    parameters = ["AWARDS", "CERTIFICATIONS", "COLLEGE NAME", "COMPANIES WORKED AT", "PHONE NUMBER", "DEGREES",
                  "PROJECT DESCRIPTIONS", "EMAIL", "NATURAL LANGUAGES", "LINKEDIN", "NAME", "SKILLS", "WORKED AS",
                  "YEAR OF GRADUATION", "YEARS OF EXPERIENCE", "RESEARCH PAPERS", "ADDRESS", "DATE OF BIRTH"]
    json_keys = ["awards", "certifications", "college_name", "companies_worked_at", "phone_number", "degrees",
                 "project_descriptions", "email", "natural_languages", "linkedin", "name", "skills", "worked_as",
                 "year_of_graduation", "years_of_experience", "research_papers", "address", "date_of_birth"]

    prompt = f"""
Extract details from the resume, which is delimited with triple backticks?

The details to be extracted are: {parameters}

Format the output as JSON object with the following keys: {json_keys}

Each key should contain the values as a list of strings.

If a specific information is not found in the resume, then set the value in that information's key as \"Not found\".

Resume: '''{text}'''
    """

    return prompt


def get_response_llama(prompt):
    model_version = "meta/llama-2-70b-chat:4dfd64cc207097970659087cf5670e3c1fbe02f83aa0f751e079cfba72ca790a"
    output = replicate.run(
        model_version=model_version,
        input={"prompt": prompt,
               "temperature": 0.01,
               "max_new_tokens": 512,
               }
    )

    response = ""

    for item in output:
        response += item

    return response


def get_response_chatgpt(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def parse_job_desc(desc, model="./Resources/Models/output/model-best"):
    nlp = spacy.load(model)
    doc = nlp(desc)

    # for ent in doc.ents:
    #     print(ent.text + " -> " + ent.label_)


def test_extract_data():
    test_str = '''
    {
    "awards": ["Runner up in Datathon", "Semi-finalist in Toycathon", "Silver Medalist in Majlish"],
    "certifications": [],
    "college_name": "Manipal University Jaipur",
    "companies_worked_at": ["MUJ ACM SIGAI Student Chapter", "Randomize MUJ", "VRPlaying Games"],
    "phone_number": "8809377988",
    "degrees": ["B.Tech in Computer Science and Engineering"],
    "project_descriptions": ["Sightscope", "Lala Arnav Vatsal", "Rainfall Prediction using RandomForest", "Project Silver Valley", "Castle Siege", "Attendance Bot"],
    "email": "arnav.vatsal2213@gmail.com",
    "natural_languages": [],
    "linkedin": [],
    "name": "Arnav Vatsal",
    "skills": ["Unity", "Python", "C#", "C++", "Artificial Intelligence", "Machine Learning", "Leadership", "Teamwork"],
    "worked_as": ["Technical Head", "Event Coordinator", "Game Developer", "Data Analyst Intern"],
    "year_of_graduation": "2026",
    "years_of_experience": "Not found",
    "research_papers": [],
    "address": "Jaipur, Rajasthan",
    "date_of_birth": "Not found"
    }
    '''
    obj = json.loads(test_str)
    return obj


def generate_questions_prompt(data):
    prompt = f'''
Generate 10 questions based on details of the candidate, provided as json which is delimited by triple backticks.
The questions should be designed to be challenging but fair.
The questions should allow the interviewer to get a good understanding of the candidate's skills and abilities, without overwhelming them or making them feel uncomfortable.
The questions should provide a better insight of the candidate's capabilities.
The questions should build up on the information provided in the resume.
The questions should check the legitimacy of the data provided.
The questions should be based on industry standards.
Format the output as json object with key: questions.

```
{data}
```
    '''
    return prompt


def analyze_resume_prompt(text, desc):
    prompt=f'''
Give scores out of 10 to the resume, delimited by triple backticks, based on how well it matches the job description, \
delimited by triple backticks.
The following factors in the resume should be considered: keywords, skills, soft skills, experience, \
projects, education.
The scores should be based on the overall match between the resume and the job description.
A higher score indicates a better match.
Provide a descriptive reason for each score.

Resume:
```
{text}
```

Job description:
```
{desc}
```

Format the output as JSON object with the following keys: scores: list of scores, reasons: list of reasons
    '''
    return prompt


def run():
    resume_file = 'Resources/Sample Resumes/LalaArnavVatsal.pdf'
    text = get_text_from_pdf(resume_file)
    file = open("Resources/Sample Outputs/resume_text.txt", 'w')
    file.write(text)
    file.close()
    data_dict, data_str = extract_data(text)

    _, data_lg = extract_data(text, model='en_core_web_lg')
    file_lg = open("Resources/Sample Outputs/resume_data_lg.txt", 'w')
    file_lg.write(data_lg)
    file_lg.write("\n\nrunning on resume parser model\n\n")
    _, data_lg = extract_data(data_lg)
    file_lg.write(data_lg)
    file_lg.close()

    _, data_md = extract_data(text, model='en_core_web_md')
    file_md = open("Resources/Sample Outputs/resume_data_md.txt", 'w')
    file_md.write(data_md)
    file_md.write("\n\nrunning on resume parser model\n\n")
    _, data_md = extract_data(data_md)
    file_md.write(data_md)
    file_md.close()

    _, data_sm = extract_data(text, model='en_core_web_sm')
    file_sm = open("Resources/Sample Outputs/resume_data_sm.txt", 'w')
    file_sm.write(data_sm)
    file_sm.write("\n\nrunning on resume parser model\n\n")
    _, data_sm = extract_data(data_sm)
    file_sm.write(data_sm)
    file_sm.close()

    file = open('Resources/Sample Job Descriptions/job_desc.txt', 'r')
    desc = file.read()
    parse_job_desc(desc=desc)
    file.close()


if __name__ == "__main__":
    resume_text = """
Summary
Passionate second-year student pursuing B.Tech CSE. Experienced in developing various projects using different
programming languages, frameworks, and tools. Proficient in competitive coding in C++. Strong interest in game
development and artificial intelligence. Dedicated to learning new technologies and skills to grow my skills.
Education
Manipal University Jaipur | Jaipur, Rajasthan
B.Tech in Computer Science and Engineering | 07/2026
Achieved a CGPA of 9.71 in the 1st year.
Delhi Public School Ranchi | Ranchi, Jharkhand
Higher Secondary School Certificate [CBSE] | 07/2022
Completed the course with a score of 89.8%
St. Thomas School Dhurwa | Ranchi, Jharkhand
Secondary School Certificate [ICSE] | 07/2020
Completed the course with a score of 95.2%
Skills
Unity, Python, C#, C++, Artificial Intelligence, Machine Learning, Leadership, Teamwork
Experience
MUJ ACM SIGAI Student Chapter | Jaipur, Rajasthan
Technical Head | 04/2023 - Present
Responsible for identifying and prioritizing technical goals for the chapter. Overseeing the implementation of cutting-edge
programs and projects. Collaborating closely with the talented and dedicated members of the chapter, including the Executive
Committee.
Randomize MUJ | Jaipur, Rajasthan
Event Coordinator | 01/2023 - 04/2023
Planning and organizing various events and activities, such as workshops, quizzes, etc. Marketing and promoting events and
activities using various channels and strategies.
VRPlaying Games
Game Developer | 09/2022 - 10/2022
I was responsible for converting pre-existing games written using Cocos2dx with C++ and Javascript to a Unity engine game
using C#. Took complete ownership of the entire codebase and ensured that the games maintained their original functionality,
performance, and quality. Implemented new features and enhancements using Unity's tools and frameworks.
DevTown (ShapeAI)
Data Analyst Intern | 06/2021 - 08/2021
Took full ownership of the product life cycle. Understood customer needs through research and market data. Owned and
shaped the backlog, roadmap, and vision of one cross-functional product team.
Projects
Sightscope
It utilizes cutting-edge artificial intelligence (AI) technology to automatically generate descriptive captions for images.
Through training an AI model on extensive datasets, we are able to extract visual features and generate meaningful
descriptions that enhance visual accessibility. Additionally, the potential applications of this project extend beyond enhancing
visual accessibility and can be utilized in areas such as surveillance and other relevant domains.
Lala Arnav Vatsal
8809377988 | arnav.vatsal2213@gmail.com | Jaipur, Rajasthan
Rainfall Prediction using RandomForest
It is a rainfall data processing model using Random Forest Regression, an ensemble learning technique that harnesses the
power of multiple decision trees. It is designed to provide forecasts on precipitation amounts for specific subdivisions within
the Indian peninsula, for both monthly and yearly intervals.
Project Silver Valley
A sci-fi first-person shooter game with complex enemy AI, full-body character animation, and dynamic shooting mechanics. I
used assets from Unity Asset Store to create a detailed game environment and incorporated APIs and SDKs for login and
leaderboard functionality.
Castle Siege
I designed a third-person multiplayer game with a real-time simulation of thousands of bots. I used advanced networking
solutions to support seamless multiplayer interactions and implemented a robust system for managing player interactions
and progress in an open-world environment.
Attendance Bot
I developed a bot for video conferencing softwares which used face recognition to mark attendance of the participants, using
Python and OpenCV library.
Awards
Runner up in Datathon
A data-focused hackathon which involved analyzing large sets of data and developing models or solutions to solve a specific
problem or challenge.
Semi-finalist in Toycathon
A national game-designing competition that challenges India s innovative minds to conceptualize novel toys and games
based on Bharatiya civilization, history, culture, mythology, and ethos.
Silver Medalist in Majlish 
An interstate sit-and-draw competition held on a specific theme. Aimed at promoting creativity and artistic skills among them.
Qualified Zonal Informatics Olympiad
Part of the Indian Computing Olympiad (ICO), it tests students  knowledge of computer science and programming.
Community Service
Fundraiser and Event Organizer
Youth Empowerment Foundation
Raised funds to distribute food packets and blankets to help the underprivileged community.
Health counselor [HIV Project]
Association of India for National Advancement
Provided HIV patients motivation and support to promote a healthy mental and physical state.
        """
    job_desc='''
We are seeking a highly motivated and talented Machine Learning Engineer Intern to join our dynamic team at Hurrey, Bengaluru.
This internship offers a unique opportunity to work on cutting-edge machine learning projects and gain hands-on experience in data science and analytics.

Responsibilities
- Research, modify, and apply data science and data analytics prototypes.
- Create and construct methods and plans for machine learning.
- Employ test findings to perform statistical analysis and improve models.
- Search the internet for readily available training datasets.
- Train and retrain ML systems and models as necessary.
- Improve and expand current ML frameworks and libraries.
- Develop machine learning applications in accordance with client or customer needs.
- Investigate, test, and implement appropriate ML tools and algorithms.
- Evaluate the application cases and problem-solving potential of ML algorithms, ranking them by success likelihood.
- Gain insights from data exploration and visualization, identifying discrepancies in data distribution that may impact model effectiveness in practical situations.

Requirements
- Currently pursuing a Bachelor's or Master's degree in Computer Science, Data Science, or a related field.
- Strong interest in machine learning, data science, and analytics.
- Basic knowledge of machine learning frameworks and libraries.
- Proficiency in programming languages such as Python or R.
- Excellent problem-solving and analytical skills.
- Strong communication and teamwork abilities.
    '''
    # response, data = extract_data_chatgpt(resume_text)
    # _, questions = generate_questions_chatgpt(data)
    _, data = analyze_resume_chatgpt(resume_text, job_desc)
    print(data)
