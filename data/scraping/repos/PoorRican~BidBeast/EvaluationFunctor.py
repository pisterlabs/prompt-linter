import warnings
from copy import copy
from typing import ClassVar

from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate, FewShotPromptWithTemplates, BasePromptTemplate

from db import SUPABASE
from functors.EmbeddingManager import EmbeddingManager
from models import Viability, FeedbackModel

FEEDBACK_PARSER = PydanticOutputParser(pydantic_object=FeedbackModel)


def process_past_jobs(jobs: list[dict]) -> list[dict]:
    processed = []
    for job in jobs:
        _job = copy(job)
        # convert viability value to text
        viability = job['viability']
        if viability == Viability.DISLIKE:
            _job['viability'] = 'not bid'
        elif viability == Viability.LIKE:
            _job['viability'] = 'bid'
        elif viability == Viability.NULL:
            warnings.warn("Encountered ambiguous viability. Skipping...")
            continue
        else:
            raise ValueError("Invalid value for 'viability'")

        if not job['reviewed']:
            warnings.warn("Encountered unreviewed row. Skipping...")
            continue

        # expand pros/cons to text list
        for kind in ('pros', 'cons'):
            reasons = ''
            for reason in job[kind]:
                reasons += f"- {reason}\n"
            _job[kind] = reasons

        processed.append(_job)

    return processed


def _evaluation_prompt(examples: list[dict]) -> BasePromptTemplate:
    _prefix = PromptTemplate.from_template("""
    You're an expert consultant assisting a freelance contractor to filter job listings on a freelancing website that
    are worthwhile to place bids on.
    
    You will be given past jobs that the freelancer has decided to bid on or has rejected. Your job is to evaluate if
    the job description is a good fit, given the skills of the freelancer, the nature of the job, and the perceived
    attributes of the prospective client. The past jobs will include a summary of what the requirements were, why the
    freelancer liked or disliked about the requirements, and if the freelancer bid on the job or not.
    
    # Past jobs:
    """)

    _suffix = PromptTemplate(template="""
    Given the feedback from past jobs, evaluate if this next job description is suitable for the freelancer based on the
    nature of the job and the expected outcomes. If the job is a good fit, `viability` should be `1`, otherwise if the
    job description is clearly not a good fit, `viability` is `0`. If you're unsure if the freelancer would like to bid
    on this job, `viability` is `-1`. Do not assume that the freelancer will like or dislike the job if the new job
    description is unlike the examples provided. 
    
    {{format_instructions}}
   
    # New Job Description:\n{{desc}}
    
    """, partial_variables={'format_instructions': FEEDBACK_PARSER.get_format_instructions()},
                             input_variables=['desc'], template_format='jinja2')

    _example_prompt = PromptTemplate.from_template("""
    ## {{title}}
    
    ### Summary
    
    {{summary}}
    
    ### Appealing Aspects of Job Requirements
    
    {{pros}}
    
    ### Unappealing Aspects of Job Requirements
    
    {{cons}}
    
    ### Viability
    
    This job was {{viability}} by the freelancer.
    """, template_format='jinja2')
    return FewShotPromptWithTemplates(prefix=_prefix,
                                      example_prompt=_example_prompt,
                                      examples=process_past_jobs(examples),
                                      suffix=_suffix,
                                      input_variables=['desc'],
                                      output_parser=FEEDBACK_PARSER,
                                      template_format='jinja2')


class EvaluationFunctor:
    manager: ClassVar[EmbeddingManager] = EmbeddingManager()

    @staticmethod
    def _preprocess_description(text: str) -> str:
        """ Preprocess job description before generating embeddings. """
        raise NotImplementedError

    @staticmethod
    def _fetch_related_rows(related_ids: list[str]):
        rows = SUPABASE.table('potential_jobs') \
            .select('summary, title, pros, cons, viability, reviewed') \
            .in_('id', related_ids).execute()
        return rows.data

    @staticmethod
    async def _process_desc(desc: str, examples: list[dict]) -> FeedbackModel:
        _prompt = _evaluation_prompt(examples)
        llm = OpenAI(temperature=0.2, model_name='gpt-3.5-turbo-16k')

        prompt = _prompt.format_prompt(desc=desc).to_string()
        response = await llm.apredict(prompt)
        return FEEDBACK_PARSER.parse(response)

    @classmethod
    async def __call__(cls, desc: str) -> FeedbackModel:
        # TODO: preprocess desc before query
        related = cls.manager.query(desc)
        examples = cls._fetch_related_rows(related)

        return await cls._process_desc(desc, examples)
