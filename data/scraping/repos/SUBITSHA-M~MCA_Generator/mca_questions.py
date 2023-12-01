import os
import re
import json
import openai
# To help construct our Chat Messages
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

# We will be using ChatGPT model (gpt-3.5-turbo)
from langchain.chat_models import ChatOpenAI

# To parse outputs and get structured data back
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

# Enter your API Key
os.environ["OPENAI_API_KEY"] = "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"

# prompt
text = """
The Lancefield-Cobaw Croziers Track planned burn was ignited on Wednesday 30 September 2015 in the Macedon Ranges Shire in spring 2015. 
The fires breached containment lines on Saturday 3 October and was brought under control overnight with approximately 70 additional hectares burnt. 
Further breaches of containment lines occurred on Tuesday 6 October and control of the bushfire was transferred from the Midlands District to the Gisborne Incident Control Centre (ICC) that afternoon. 
The bushfire, when finally contained on Tuesday 13 October, had burnt over 3,000ha and destroyed several dwellings, numerous sheds and many kilometres of fencing. 
It had also impacted upon lifestyles, livestock and livelihoods and caused considerable economic and social upheaval in the surrounding communities.
"""

from transformers import pipeline
def get_mca_questions(text:str)->list:
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    print(summary[0]['summary_text'])
    response_schemas = [
    ResponseSchema(name="question", description="A multiple choice question generated from input text snippet."),
    ResponseSchema(name="options", description="Four choices for the multiple choice question."),
    ResponseSchema(name="answer", description="Two Correct answers for the question.")
]

# The parser that will look for the LLM output in my schema and return it back to me
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# The format instructions that LangChain makes. Let's look at them
    format_instructions = output_parser.get_format_instructions()

    print(format_instructions)
# create ChatGPT object
    chat_model = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
# The prompt template that brings it all together
    prompt = ChatPromptTemplate(
        messages=[
        HumanMessagePromptTemplate.from_template("""Generate multiple-choice questions (MCQs) with four options for each question. Please ensure that two of the options are correct answers and also each question must have serial number . You should Provide the output in the form of a list. 
        \n{format_instructions}\n{user_prompt}""")
    ],
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions}
    )
    user_query = prompt.format_prompt(user_prompt = summary)
    user_query_output = chat_model(user_query.to_messages())
    print(user_query_output)
    markdown_text = user_query_output.content
    json_string = re.search(r'{(.*?)}', markdown_text, re.DOTALL).group(1)
    print(json_string)
    options_list = json_string.split('\n')
    question = options_list[0].strip()
    answer = options_list[-1].strip()
    options = [opt.strip() for opt in options_list[1:-1] if opt.strip()]
    question_data = [question] + options + [answer]
    print(question_data)
    return question_data
get_mca_questions(text)
