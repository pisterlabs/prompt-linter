import typing
from keytotext import trainer
from huggingface_hub import Repository
import torch

import os
from langchain.llms import OpenAI
from langchain import PromptTemplate, FewShotPromptTemplate, LLMChain

from dotenv import load_dotenv

load_dotenv()


def download_model():
    """ 
        Loads the company description finetuned T5 model: 
        >>> model = download_model()
        >>> descriptions = model.predict(keywords=body.keywords, use_gpu=True)
    """
    model = trainer()

    model_repo = Repository(
        "model",
        "ashrielbrian/t5-base-wikipedia-companies-keywords",
        token=os.getenv("HUGGINGFACE_API_TOKEN"),
        git_user="ashrielbrian",
    )

    use_gpu = False
    # check if gpu is available
    if torch.cuda.is_available():
        use_gpu = True

    model.load_model(model_dir="model", use_gpu=use_gpu)
    return model


# output = model.predict(keywords=["band and roll", "german", "leather goods", "iphone", "ipad"], use_gpu=True)
# print(output)

def get_few_shot_prompt(fp: str):
    """ Loads the prompt templates for company descriptions"""
    import json

    examples = json.load(open(fp, 'r'))

    example_format_template = """
        Company: {company}
        Keywords: {keywords}
        Description: {description}
    """

    example_prompt = PromptTemplate(template=example_format_template, input_variables=["company", "keywords", "description"])

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""I want you to act as a Search Engine Optimization (SEO) consultant to generate new company descriptions.
        Here are some examples of good SEO-optimized company descriptions.""",
        suffix="""Company: {company}
        Keywords: {keywords}
        Description:
        """,
        example_separator="\n\n",
        input_variables=["company", "keywords"]
    )

    return few_shot_prompt

def load_descriptor(fp: str):
    """ Loads the LLM Chain to generate descriptive text"""
    llm = OpenAI(model_name="gpt-3.5-turbo")
    prompt = get_few_shot_prompt(fp)
    return LLMChain(prompt=prompt, llm=llm)

def get_company_description(llm_chain: LLMChain, company_name: str, keywords: typing.List[str]):
    return llm_chain.run(company=company_name, keywords=", ".join(keywords))


