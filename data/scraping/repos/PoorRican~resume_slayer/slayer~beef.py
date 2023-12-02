""" Chains for improving a job experience section when it is not strong for a given job description / title """

from langchain import LLMChain, PromptTemplate
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI

from .util import relevant_skills_chain


def _summary_sentence_from_skills() -> LLMChain:
    """ Generate a summary sentence showing how a job experience is demonstrates the given skills.

    This produces a nicely worded, concise summary. This might be useful in the future,
    but is not planned in be implemented now.

    Inputs:
    skills: list of 3 requirements extracted from job description. eg: `util.job_requirements_chain()`
    section: job history section from resume

    Outputs:
    a 1-sentence summary than nicely describes the job history using given points. This summary omits anything
    that does not pertain to skills.
    """
    prompt = PromptTemplate.from_template("""
    You will be given a section of a resume and 3 key skills.
    Please write a one sentence summary for the resume section highlighting some or all of the given skills.
    
    Skills: {skills}
    
    Section: {section} 
    """)
    return LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=.1, model_name="gpt-3.5-turbo"),
                    output_key='summary')


def highlight_chain() -> LLMChain:
    """ Frame weak job experience section to highlight key skills.

    Notes:
    - Structures the entire job history section under the given 3 skills.
    - Might be wordy, but tends to include more details than `_summary_sentence_from_skills()`

    Inputs:
    skills: list of 3 requirements extracted from job description. eg: `util.job_requirements_chain()`
    section: job history section from resume

    Outputs:
    Job history section is restructured under the 3 key skills as bullet points. Most of the original data is retained.
    """

    prompt = PromptTemplate.from_template("""
    You will be given a job experience from a resume and 3 key skills.
    Use bullet points to highlight how the job experience section demonstrates the given skills.
    
    Skills: {skills}
    
    Job Experience: {section} 
    """)
    return LLMChain(prompt=prompt,
                    llm=ChatOpenAI(temperature=.1, model_name="gpt-3.5-turbo"),
                    output_key="highlighted")


def format_chain() -> LLMChain:
    # chain to format section as markdown
    format_prompt = PromptTemplate.from_template("""
    May you please reformat the following experience from a resume using the following format:

    ```
    ## Job Title, Company, Dates (Total time)

        - bullet points
    ```

    Here is experience text:
    \n{highlighted}
    """)
    return LLMChain(prompt=format_prompt,
                    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
                    output_key="formatted")


def process_history_chain() -> SequentialChain:
    # chain emulate section wording and grammar
    gram_prompt = PromptTemplate.from_template("""
        You're an expert career consultant with an IQ over 140 working with a special client regarding this job posting.
        Please improve this resume section for this {title} position.\n
        Improve the section by matching grammatical syntax and lexicon.

        This is the job description:\n\n{desc}.
        \n\n
        Here is the resume section:\n{section}
    """)
    grammatical_chain = LLMChain(prompt=gram_prompt,
                                 llm=ChatOpenAI(temperature=.85, model_name="gpt-3.5-turbo"),
                                 output_key="emulated"
                                 )

    return SequentialChain(
        chains=[grammatical_chain, format_chain()],
        input_variables=["title", "desc", "section"],
        output_variables=["formatted"],
        verbose=True,)


def beef_chain() -> SequentialChain:
    """ Improve a weak job experience section by emphasizing relevant skills

    Inputs:
    title: desired job title
    desc: desired job description
    requirements: desired job requirements. From `util.job_requirements_chain()`
    section: job experience section from resume. metadata will NOT be formatted (ie: dates, job title, etc.)

    Outputs:
    Job experience section highlighting relevant job requirements
    """
    return SequentialChain(
        chains=[relevant_skills_chain(), highlight_chain()],
        input_variables=["title", "desc", "section", "requirements"],
        output_variables=["highlighted"],
        verbose=True
    )
