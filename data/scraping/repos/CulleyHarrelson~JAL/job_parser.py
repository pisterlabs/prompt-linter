import requests
import constants

# import openai
# from langchain.agents import AgentType, initialize_agent , load_tools
from langchain.llms import OpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


def lookup_job(url):
    urls = [url]
    loader = UnstructuredURLLoader(urls=urls)
    jd = loader.load()
    return jd


def parse_job_description(url):
    model_name = "text-davinci-003"
    temperature = 0.0
    model = OpenAI(model_name=model_name, temperature=temperature)

    class JobApplication(BaseModel):
        company_name: str = Field(
            description="The name of the company in the job description"
        )
        job_title: str = Field(description="The job title in the job description")
        low_salary_range: str = Field(
            description="Given a salary range for the job description, this is the lower number"
        )
        high_salary_range: str = Field(
            description="Given a salary range for the job description, this is the greater number"
        )

    # And a query intented to prompt a language model to populate the data structure.
    jd = lookup_job(url)

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=JobApplication)

    prompt = PromptTemplate(
        template="Parse the following online job description \n{format_instructions}\n{job_description}\n",
        input_variables=["job_description"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    _input = prompt.format_prompt(job_description=jd)

    output = model(_input.to_string())

    return parser.parse(output)


# contains an iframe, cant find location
# jd = url_to_plaintext('https://c3.ai/job-description/?gh_jid=4112793002')


if __name__ == "__main__":
    # execute only if run as a script
    url = "https://careers.lamresearch.com/job/Data-Security-Engineer/1042795300/"
    print(parse_job_description(url=url))

    # TODO - string to long
    # url2 = "https://c3.ai/job-description/?gh_jid=4112793002"
    # print(parse_job_description(url=url2))

# jd
