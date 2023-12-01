from typing import List
from hr_job_cv_matcher.service.test.job_description_cv_provider import (
    job_description_cv_provider,
)

from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain import LLMChain
from langchain.chains.openai_functions import create_structured_output_chain

from hr_job_cv_matcher.service.job_description_cv_matcher import (
    JOB_DESCRIPTION_START,
    JOB_DESCRIPTION_END,
    CV_START,
    CV_END,
    HUMAN_MESSAGE_JD,
    HUMAN_MESSAGE_CV,
    TIPS_PROMPT,
    prompt_factory,
)

from hr_job_cv_matcher.config import cfg


HR_SYSTEM_MESSAGE = "You are an expert in human resources and you are an expert at extracting relevant job information and degrees from a CV based on a job description."
HUMAN_MESSAGE_1 = f"""Please extract first the job titles from the CV (curriculum vitae of a candidate). The extracted job titles should be relatd to the job description.
The job description part starts with {JOB_DESCRIPTION_START} and ends with {JOB_DESCRIPTION_END}.
The CV (curriculum vitae of a candidate) description part starts with {CV_START} and ends with {CV_END}. Then output the matching, missing and associated skills using the provided JSON structure."""


class EducationCareer(BaseModel):
    """Contains informations about the education and jobs of a candidate"""

    relevant_job_list: List[str] = Field(
        ...,
        title="Job list",
        description="The list of jobs of a candidate which are mentioned in the text and that are related to the job description",
    )
    relevant_degree_list: List[str] = Field(
        ...,
        title="Degree list",
        description="The list of degrees of a candidate which are mentioned in the text and that are related to the job description",
    )


json_schema_education_extraction = {
    "title": "EducationCareer",
    "description": "Collects the relevant job list and the degree list of a candidate for a job specification",
    "type": "object",
    "properties": {
        "relevant_job_list": {
            "title": "Relevant jon list",
            "description": "The list of jobs of a candidate which are mentioned in the text and that are related to the job description",
            "type": "array",
            "items": {"type": "string"},
        },
        "relevant_degree_list": {
            "title": "Relevant degree list",
            "description": "The list of degrees of a candidate which are mentioned in the text and that are related to the job description",
            "type": "array",
            "items": {"type": "string"},
        },
        "years_of_experience": {
            "title": "Total years of professional experience",
            "description": "The total years of professional experience of a candidate. This information comes from the CV of the candidate.",
            "type": "integer"
        }
    },
    "required": ["relevant_job_list", "relevant_degree_list", "years_of_experience"],
}


def create_zero_shot_matching_prompt() -> ChatPromptTemplate:
    return prompt_factory(
        HR_SYSTEM_MESSAGE,
        [HUMAN_MESSAGE_1, HUMAN_MESSAGE_JD, HUMAN_MESSAGE_CV, TIPS_PROMPT],
    )


def create_education_chain() -> LLMChain:
    return create_structured_output_chain(
        json_schema_education_extraction, cfg.llm, create_zero_shot_matching_prompt(), verbose=True
    )


def create_education_chain_pydantic() -> LLMChain:
    return create_structured_output_chain(
        json_schema_education_extraction, cfg.llm, create_zero_shot_matching_prompt(), verbose=True
    )


if __name__ == "__main__":
    job_description, cvs = job_description_cv_provider()
