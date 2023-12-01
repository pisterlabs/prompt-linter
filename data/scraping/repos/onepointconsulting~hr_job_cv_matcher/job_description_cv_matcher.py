import re
from typing import List
from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains.openai_functions import create_structured_output_chain

from hr_job_cv_matcher.log_init import logger
from hr_job_cv_matcher.config import cfg, prompt_cfg


HR_SYSTEM_MESSAGE = "You are an expert in human resources and you are an expert at matching skills from a job description to a CV of a candidate"
JOB_DESCRIPTION_START = "=== 'JOB DESCRIPTION:' ==="
JOB_DESCRIPTION_END = "=== 'END JOB DESCRIPTION' ==="
CV_START = "=== CV START: ==="
CV_END = "=== CV END: ==="
JOB_DESCRIPTION_KEY='job_description'
PLACE_HOLDER_JOB_DESCRIPTION = f"{{{JOB_DESCRIPTION_KEY}}}"
PLACE_HOLDER_KEY = "cv"
PLACE_HOLDER_CV = f"{{{PLACE_HOLDER_KEY}}}"
HUMAN_MESSAGE_1 = f"""Please extract first the skills from the job description. 
The job description part starts with {JOB_DESCRIPTION_START} and ends with {JOB_DESCRIPTION_END}.
The CV (curriculum vitae of a candidate) description part starts with {CV_START} and ends with {CV_END}. 
Then output the matching, missing and associated skills using the provided JSON structure.
The matching skills are the skills in the job description part which are also found in the CV (curriculum vitae of a candidate) description part.
The missing skills are those skills which are in the jobo description part but not in the CV (curriculum vitae of a candidate) description part.

Here are some examples of skills that you might find in the job descriptions and CVs:
- Wordpress
- Creating Wordpress websites
- Website Optimization
- PHP
- SQL
- Javascript
- Debugging
- HTML
- HTML5
- CSS
- CSS3
- WOO-Commerce Management
- Client Support
- Python 
- Linux, macOS, and Windows
- Git
- Building E-commerce stores using woocmmerce plugin
- Front-end development
- Codeigniter
- Programming languages: C, C++
- Machine Learning
- Deep Learning
- Database: MySQL
- Database: MongoDB
- IDEs: IntelliJ
- Azure Logic apps
- Azure Data Factor
- Azure Functions
- Experience with REST APIs
- Experience with Business Intelligence BI
- Analytical reporting using PowerBI
- Exposure to ITIL
{{extra_skills}}

Here is an example of how you extract matching and missing skills:

====== Example start: ======
{JOB_DESCRIPTION_START}
Application Support Analyst (L2/L3) at Onepoint (PUNE_JD2023-02_IN.007)

Minimum Bachelor's Degree in Computer Science or equivalent stream.

2-3 years experience in Cloud-based (Azure, AWS and Google Cloud) Application Support,
ideally ina Managed service environment..

Experience in supporting core business applications on Azure (Logic apps, Data Factory,
Functions) and Snowflake or equivalent cloud data warehouse.

Experience with REST APls, SQL, JSON, XML.

Experience with Business Intelligence BI and Analytical reporting using PowerBI.
Knowledge in programming languages like Python, Java.

Knowledge about integration platforms or low code platforms.

Exposure to ITIL.

Knowledge in Al, ML preferred.

Excellent written and verbal communication skills in English.

Excellent interpersonal skills to collaborate with various stakeholders.

A learning enthusiast who would quickly pick up new programming languages, technologies,
and frameworks.

A proactive Self-Starter with excellent time management skills.

Problem-solving and analytical skills across technical, product, and business questions.
{JOB_DESCRIPTION_END}

{CV_START}

Experienced web developer with a proven track record of creating dynamic and user-friendly websites. Proficient
in various programming languages and frameworks, with a strong emphasis on front-end development.
Committed to delivering high-quality code and exceptional user experiences. Strong problem-solving and
communication skills, with the ability to collaborate effectively in cross-functional teams.

Key Skills: HTML5, CSS3, JavaScript, Bootstrap, Responsive Design, UX/UI, RESTful APIs, JSON, XML, Git, Agile
Development, SEO Optimization, Performance Optimization, Cross-Browser Compatibility, Testing and
Debugging, WordPress, PHP, MySQL.

Profile Summary: Highly skilled web developer with expertise in HTML5, CSS3, and JavaScript.Focus on
building responsive and intuitive user interfaces. Strong understanding of UX/UI principles and ability to optimize
websites for performance and SEO. Experienced in working with RESTful APIs and version control systems like
Git. Collaborative team player with excellent problem-solving abilities and a passion for staying up-to-date with
the latest industry trends and technologies. Extensive experience in WordPress development and familiarity with
back-end technologies like PHP, MySQL and SQL. Effective communicator with a proven ability to work in fast-paced,
Agile environments.
{CV_END}

The matching skills are: RESTful APIs, SQL, JSON, XML
The missing skills are: Cloud-based Application Support, integration platforms, Business Intelligence BI, Analytical reporting using PowerBI, Exposure to ITIL, Knowledge in Al, Azure Logic apps, Azure Data Factory

====== Example end: ======

"""
HUMAN_MESSAGE_JD = f"""{JOB_DESCRIPTION_START}
{PLACE_HOLDER_JOB_DESCRIPTION}
{JOB_DESCRIPTION_END}
"""
HUMAN_MESSAGE_CV = f"""{CV_START}
{PLACE_HOLDER_CV}
{CV_END}
"""
TIPS_PROMPT = "Tips: Make sure you answer in the right format"


class MatchSkillsProfile(BaseModel):
    """Contains the information on how a candidate matched the profile."""

    matching_skills: List[str] = Field(..., description="The list of skills of the candidate which matched the skills in the job description.")
    missing_skills: List[str] = Field(..., description="The list of skills that are in the job description, but not matched in the job profile.")
    social_skills: List[str] = Field(..., description="A list of skills which are mentioned in the candidate CV only.")


json_schema_match_skills = {
    "title": "MatchingSkills",
    "description": "Collects matching and missing skills between a candidate's CV and a job application",
    "type": "object",
    "properties": {
        "matching_skills": {
            "title": "Matching skills list",
            "description": "The list of skills of the candidate which matched the skills in the job description.",
            "type": "array",
            "items": {"type": "string"},
        },
        "missing_skills": {
            "title": "Missing skills list",
            "description": "The list of skills that are in the job description, but not matched in the job profile.",
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["matching_skills", "missing_skills"],
}


def prompt_factory(system_message: str, human_messages: List[str]) -> ChatPromptTemplate:
    assert len(human_messages) > 0, "The human messages cannot be empty"
    final_human_messages = []
    count_template = 0
    regex = re.compile(r"\{[^}]+\}", re.MULTILINE)
    for m in human_messages:
        if re.search(regex, m):
            # In case there is a placeholder
            final_human_messages.append(HumanMessagePromptTemplate.from_template(m))
            count_template += 1
        else:
            # No placeholder
            final_human_messages.append(HumanMessage(content=m))
    assert count_template > 0, "There has to be at least one human message with {}"
    logger.info("Template count: %d", count_template)
    prompt_msgs = [
        SystemMessage(
            content=system_message
        ),
        *final_human_messages
    ]
    return ChatPromptTemplate(messages=prompt_msgs)


def create_zero_shot_matching_prompt() -> ChatPromptTemplate:
    system_message = HR_SYSTEM_MESSAGE
    human_message_1 = HUMAN_MESSAGE_1.format(extra_skills=prompt_cfg.extra_skills)
    logger.info("human_message_1: %s", human_message_1)
    return prompt_factory(system_message, [human_message_1, HUMAN_MESSAGE_JD, HUMAN_MESSAGE_CV, TIPS_PROMPT])


def create_match_profile_chain_pydantic() -> LLMChain:
    return create_structured_output_chain(MatchSkillsProfile, cfg.llm, create_zero_shot_matching_prompt(), verbose=cfg.verbose_llm)


def create_match_profile_chain() -> LLMChain:
    return create_structured_output_chain(json_schema_match_skills, cfg.llm, create_zero_shot_matching_prompt(), verbose=cfg.verbose_llm)


def create_input_list(job_description, cvs):
    return [{JOB_DESCRIPTION_KEY: job_description, PLACE_HOLDER_KEY: cv} for cv in cvs]

