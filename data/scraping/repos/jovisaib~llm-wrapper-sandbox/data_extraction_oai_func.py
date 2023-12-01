
from typing import Optional, List
from pydantic import BaseModel

from openai_function_call import OpenAISchema
from pydantic import Field
import openai





class Technology(BaseModel):
    name: str = Field(..., description="one specific technology (like a language, framework, platform)")
    category: str

class WorkExperience(BaseModel):
    role: str
    company: str
    start_date: str
    end_date: str
    responsibilities: List[str]

class Education(BaseModel):
    degree: str
    institution: str
    location: str
    start_date: str
    end_date: str

class Language(BaseModel):
    name: str
    level: str

class Resume(OpenAISchema):
    "Correctly extracted information from a resume"
    name: str
    contact_information: str
    objective_statement: str
    other_skills: List[str] = Field(..., description="name of a non-technical specific skill (like 'teamwork', 'empathy')")
    technologies: List[Technology]
    work_experience: List[WorkExperience]
    education: List[Education]
    certifications: List[str]
    language: List[Language]
    references: Optional[str]


def parse_resume(resume: str) -> Resume:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=0.,
        functions=[Resume.openai_schema],
        function_call={"name": Resume.openai_schema["name"]},
        messages=[
            {"role": "system", "content": "Extract all the information fot the resume provided"},
            {"role": "user", "content": resume},
        ],
    )
    return Resume.from_response(completion)


resume = """
    **John Doe**

    1234 Main Street, San Francisco, CA 94101 | (123) 456-7890 | john.doe@email.com | LinkedIn: /in/johndoe

    **Objective**

    Highly motivated Software Engineer specializing in Machine Learning, Artificial Intelligence, and MLOps, seeking to leverage extensive background in software development, data science, and AI systems in a challenging new role.

    **Skills**

    - Proficiency in Python, Java, and C++
    - Experience in Machine Learning and Deep Learning frameworks (TensorFlow, PyTorch, Keras)
    - Expertise in MLOps tools (Kubeflow, Docker, Jenkins)
    - Knowledge of cloud platforms (AWS, Google Cloud, Azure)
    - Experience with Big Data tools (Hadoop, Spark)
    - Excellent problem-solving skills and attention to detail

    **Work Experience**

    **Software Engineer, Machine Learning**
    Google, Mountain View, CA | June 2019 – Present

    - Designed and implemented ML models for various Google services, improving prediction accuracy by 20%.
    - Deployed robust ML pipelines using MLOps best practices.
    - Coordinated cross-functional teams to integrate ML models into product offerings.
    - Optimized machine learning algorithms to enhance system performance.

    **Software Engineer, AI**
    Microsoft, Redmond, WA | August 2016 – May 2019

    - Participated in the development of AI models for various Microsoft products.
    - Assisted in the establishment of the company's MLOps framework.
    - Created reusable code and libraries for future use.
    - Improved efficiency of existing systems by implementing AI solutions.

    **Junior Software Developer**
    IBM, Armonk, NY | July 2014 – July 2016

    - Developed and maintained software applications.
    - Collaborated with senior engineers to develop machine learning capabilities.
    - Assisted in bug fixing and improving application performance.

    **Education**

    **Master of Science in Computer Science (Specialization in AI)**
    Stanford University, Stanford, CA | 2012 – 2014

    **Bachelor of Science in Computer Science**
    University of California, Berkeley, CA | 2008 – 2012

    **Certifications**

    - Certified TensorFlow Developer – TensorFlow
    - Certified Professional in AI & ML – Microsoft
    - Certified Solutions Architect – Amazon Web Services

    **Languages**

    - English (native)
    - Spanish (fluent)
"""


print(parse_resume(resume).model_dump_json(indent=2))