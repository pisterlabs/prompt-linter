import os
from langchain.llms.llamacpp import LlamaCpp
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from pydantic import BaseModel, Field

from question import select_random_n_questions, generate_sample_question
from pprint import pprint
import json


class Exercise(BaseModel):
    topic: str = Field(description="The topic of the exercise")
    title: str = Field(description="The title of the exercise")
    problem_statement: str = Field(description="The question of the exercise")
    solution: str = Field(description="The sample solution of the exercise")


def get_parser(object):
    return PydanticOutputParser(pydantic_object=object)

def get_llm(model: str = "gpt-3.5-turbo",
            temperature: float = 0.8,
            tag: str = "test-run"):
    llm = ChatOpenAI(model=model,
                     temperature=temperature,
                     tags=[tag])
    return llm

def create_code_explanation_prompt(generated_question,
                                   code):
    return f"""Given the following problem statement:
{generated_question}
Given the following code that addressed the above problem statement:
{code}
"""


def create_exercise_prompt(language,
                           sample_questions,
                           topic):
    prime = """\"\"\"Exercise {num}
---Keywords---
{topic}
---Title--
{title}
---Problem Statement---
{content}
---Sample Solution---
{solution}
"""

    result_prompt = ""
    num_questions = len(sample_questions)
    for idx, val in enumerate(sample_questions):
        result_prompt += prime.format(num=idx+1,
                                      topic=topic,
                                      title=val['title'],
                                      content=val['question'],
                                      solution=val['answer'].replace(language, ""))

    result_prompt += f"\"\"\"Exercise {num_questions+1}"
    result_prompt = result_prompt.replace("{", "(").replace("}", ")")
    return result_prompt

def get_llm_chain(llm,
                  template,
                  parser=None,
                  tag='test-run'):


    if parser:
        template += """\n\n{format_instructions}
        {question}
        """
        prompt = PromptTemplate(template=template,
                                input_variables=['question'],
                                partial_variables={"format_instructions": parser.get_format_instructions()}
        )
    else:
        template += """{question}
        """
        prompt = PromptTemplate(template=template,
                                input_variables=['question'])

    llm_chain = LLMChain(llm=llm,
                         prompt=prompt,
                         tags=[tag])
    return llm_chain

def parse_response(response, parser, llm, prompt_value):
    response = response.replace("`", "")
    while True:
        try:
            generated_exercise = json.loads(response)
            return generated_exercise
        except Exception as ex:
            retry_parser = RetryWithErrorOutputParser.from_llm(
                parser=parser, llm=llm
            )
            return retry_parser.parse_with_prompt(response, prompt_value)


# if __name__ == "__main__":

#     language = 'python'
#     difficulty = 'Easy'
#     topic = 'Array'

#     dataset_path = generate_sample_question(language=language,
#                                             difficulty=difficulty,
#                                             topic=topic)
#     num_ref_exercises = 3
#     sample_questions = select_random_n_questions(dataset_path=dataset_path, n=num_ref_exercises)
#     prompt = create_exercise_prompt(sample_questions=sample_questions,
#                                     topic=topic)
#     model_path = f"{os.getcwd()}/llama.cpp/models/7b/ggml-model-q4_0.bin"
#     llm = get_llm(model_path=model_path)
#     llm_chain = get_llm_chain(llm, prompt)

#     response = llm_chain("Generate python coding exercise according to above format")
#     parse_response(response)