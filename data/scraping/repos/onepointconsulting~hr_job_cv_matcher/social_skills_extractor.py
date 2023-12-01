from typing import List

from hr_job_cv_matcher.service.test.job_description_cv_provider import CV
from langchain import LLMChain

from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from hr_job_cv_matcher.service.job_description_cv_matcher import CV_END, CV_START, PLACE_HOLDER_CV, PLACE_HOLDER_KEY
from hr_job_cv_matcher.config import cfg
from hr_job_cv_matcher.log_init import logger

from langchain.chains.openai_functions import create_structured_output_chain


HR_SYSTEM_MESSAGE = "You are an human resources expert and you are great at finding social skills"
HUMAN_MESSAGE_1 = f"""Please extract all social skills from the CV presented below: 
The CV (curriculum vitae of a candidate) description part starts with {CV_START} and ends with {CV_END}.

{CV_START}
{PLACE_HOLDER_CV}
{CV_END}

Here are some examples of skills that you might find in the job descriptions and CVs:
- Excellent written and verbal communication skills in English
- Excellent interpersonal skills to collaborate with various stakeholders
- Problem-solving and analytical skills across technical, product, and business questions
"""

def create_social_prompt():
    prompt_msgs = [
        SystemMessage(
            content=HR_SYSTEM_MESSAGE
        ),
        HumanMessagePromptTemplate.from_template(HUMAN_MESSAGE_1)
    ]
    return ChatPromptTemplate(messages=prompt_msgs)


def create_social_profile_chain() -> LLMChain:
    return create_structured_output_chain(json_schema_match_skills, cfg.llm, create_social_prompt(), verbose=cfg.verbose_llm)


json_schema_match_skills = {
    "title": "SocialSkills",
    "description": "Collects social skills mentioned in a CV",
    "type": "object",
    "properties": {
        "social_skills": {
            "title": "Social skills list",
            "description": "A list of skills which are mentioned in the candidate CV only.",
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["social_skills"],
}


def create_input_list(cvs):
    return [{PLACE_HOLDER_KEY: cv} for cv in cvs]



def extract_social_skills(result) -> List[str]:
    res = None
    if 'function' in result:
        res: dict = result["function"]
    elif 'social_skills' in result:
        res: dict = result['social_skills']
    return res


if __name__ == "__main__":
    llm_chain = create_social_profile_chain()
    res = llm_chain.run({PLACE_HOLDER_KEY: CV})
    logger.info(type(res))
    logger.info(res)
