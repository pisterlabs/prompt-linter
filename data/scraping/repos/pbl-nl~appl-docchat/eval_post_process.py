
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import AIMessage,HumanMessage
from langchain.chains import create_extraction_chain
import settings
import openai
import json
import os
import pandas as pd
load_dotenv()

# The following assumes that the JSON file and .tsv files are aligned (i.e., 
# that a full evaluation has taken place based on all input in the JSON file)
# and relevant .TSV files are output

def compile_prompt(query_text:str)->list:
    """Given the question, compile the prompt to categorise question type 
    information from the text."""
    sys_prompt = SystemMessagePromptTemplate.from_template(system)
    human_eg_template = HumanMessage(content=human_eg, example=True)
    ai_template = AIMessage(content=ai_eg, example=True, additional_kwargs={"validate":False})
    human_prompt = HumanMessagePromptTemplate.from_template('{query_text}')
    full_prompt = ChatPromptTemplate.from_messages(
        [sys_prompt, human_eg_template, ai_template, human_prompt]
    )
    # last step is to provide query text and format instructions
    complete_prompt = full_prompt.format_prompt(
        formatting_instructions = format_instructions,
        query_text = query_text
    ).to_messages()
    return complete_prompt

# Specify question types
type_1 = "Type 1: Generic content queries. Examples: 'What are the authors names?', 'What year was the report or article published?'"
type_2 = "Type 2: Summary queries. Examples: 'Give me a summary of the report content.', 'Provide a bullet point list of the main points.', 'Was the study succesful?'"
type_3 = "Type 3: Specific queries. Examples: 'What was the parameter value for x.', 'How many years did the model run for?., 'Which country was involved in x?'"
type_4 = "Type 4: Example queries. Examples: 'Give me an example case of when the model was succesful.', 'Provide an example in the text that demonstrates x'."

# Formatting instructions for prompt
format_instructions = "Only return either 1, 2, 3, 4 as an integer only according to question types"
system = fr"You job is to classify questions according to four types: \n {type_1}, {type_2}, {type_3}, {type_4}"
human_eg = "What part of Europe was the study undertaken?"
ai_eg = 'The question type is 2'

# load available evalutation result names based on JSON file
with open(os.path.join(settings.EVAL_DIR, settings.EVAL_FILE_NAME), 'r') as eval_file:
    eval = json.load(eval_file)

# Loop over question types and categorise them
for folder in eval:
    categories = []
    path = os.path.join(settings.EVAL_DIR, f"{folder}.tsv")
    query = pd.read_csv(path, delimiter = "\t")
    for question in query.question:
        prompt = compile_prompt(question)
        chat = ChatOpenAI(temperature = 0.1)
        result = chat.invoke(prompt)
        result = result.content
        result = int("".join([i for i in result if i.isdigit()]))
        categories.append(result)
        print(question)
        print("Question category:", result)
    query['question_category'] = categories
    query.to_csv(path, sep="\t", index=False)
