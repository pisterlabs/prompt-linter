#### This file is an incomplete attempt to create a chain which extracts first the skills and then 
#### used a subsequent prompt to find the matching and missing skills. This is not used right now.
from typing import List, Optional
from hr_job_cv_matcher.service.test.job_description_cv_provider import app_support_analyst_provider
from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains.openai_functions import create_structured_output_chain

from hr_job_cv_matcher.log_init import logger
from hr_job_cv_matcher.config import cfg


HR_SYSTEM_MESSAGE_SKILLS = "You are an expert in human resources and you are an expert at extracting technical skills from job descriptions"
HR_SYSTEM_MESSAGE_MATCHING_SKILLS = "You are an expert in human resources and you are an expert at matching skills"
JOB_DESCRIPTION_START = "=== 'JOB DESCRIPTION:' ==="
JOB_DESCRIPTION_END = "=== 'END JOB DESCRIPTION' ==="
CV_START = "=== CV START: ==="
CV_END = "=== CV END: ==="
JOB_DESCRIPTION_KEY='job_description'
PLACE_HOLDER_JOB_DESCRIPTION = f"{{{JOB_DESCRIPTION_KEY}}}"
PLACE_HOLDER_KEY = "cv"
PLACE_HOLDER_CV = f"{{{PLACE_HOLDER_KEY}}}"
HUMAN_MESSAGE_JD_SKILLS = f"""
Please extract the skills from the job description. 
The job description part starts with {JOB_DESCRIPTION_START} and ends with {JOB_DESCRIPTION_END}.

Here are some examples of skills that you might find in the job descriptions:
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

====== Example end: ======

Please note that names of places and people are not technical skills.

{JOB_DESCRIPTION_START}
{PLACE_HOLDER_JOB_DESCRIPTION}
{JOB_DESCRIPTION_END}
"""

HUMAN_MESSAGE_MATCHING_SKILLS = f"""
Please extract the skills from a CV which match the following list of skills:
{{extracted_skills}} 
The part with the CV starts with {CV_START} and ends with {CV_END}.

====== Example end: ======

{CV_START}
{PLACE_HOLDER_CV}
{CV_END}
"""

TIPS_PROMPT = "Tips: Make sure you answer in the right format"


class SkillsProfile(BaseModel):
    """Contains the information about the skills found in a job description"""
    technical_skills: List[str] = Field(..., description="The list of technical skills which can be found in the job description.")


class MatchingSkillsProfile(BaseModel):
    """Contains the information about the skills that match the provided list of skills"""
    matching_technical_skills: List[str] = Field(..., description="The list of technical skills which match the provided skills")


def prompt_factory(system_message: str, main_human_message: str) -> ChatPromptTemplate:
    prompt_template = HumanMessagePromptTemplate.from_template(main_human_message)
    prompt_msgs = [
        SystemMessage(content=system_message),
        prompt_template,
        HumanMessage(content=TIPS_PROMPT)
    ]
    return ChatPromptTemplate(messages=prompt_msgs)


def create_skill_prompt() -> ChatPromptTemplate:
    return prompt_factory(HR_SYSTEM_MESSAGE_SKILLS, HUMAN_MESSAGE_JD_SKILLS)


def create_skill_extraction_profile_chain_pydantic() -> LLMChain:
    return create_structured_output_chain(SkillsProfile, cfg.llm, create_skill_prompt(), verbose=cfg.verbose_llm)


def create_skill_extraction_input(job_description: str, extra_skills: str = '') -> dict:
    return {JOB_DESCRIPTION_KEY: job_description, 'extra_skills': extra_skills}


def extract_skills_profile(response) -> Optional[SkillsProfile]:
    if 'function' in response:
        skills_profile: SkillsProfile = response['function']
        return skills_profile
    if isinstance(response, SkillsProfile):
        return response
    else:
        logger.warn("Could not extract from %s", response)
        return None
    

def create_matching_skills_prompt() -> ChatPromptTemplate:
    return prompt_factory(HR_SYSTEM_MESSAGE_MATCHING_SKILLS, HUMAN_MESSAGE_MATCHING_SKILLS)


def create_matching_skill_chain_pydantic() -> LLMChain:
    return create_structured_output_chain(MatchingSkillsProfile, cfg.llm, create_matching_skills_prompt(), verbose=cfg.verbose_llm)


def create_skill_extraction_input(cv: str, extracted_skills: str = '') -> dict:
    return {PLACE_HOLDER_KEY: cv, 'extracted_skills': extracted_skills}


if __name__ == "__main__":
    jd = app_support_analyst_provider()
    skill_extraction_chain = create_skill_extraction_profile_chain_pydantic()
    res = skill_extraction_chain.run(create_skill_extraction_input(jd))
    skills_profile: SkillsProfile = extract_skills_profile(res)
    logger.info("Skill extraction result: %s", skills_profile)
    logger.info("Skill extraction result type: %s", type(res))


