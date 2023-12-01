from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import dotenv_values
import os 
import streamlit as st 
import openai 

openai.api_base = "https://oai.hconeai.com/v1"

# Load environment variables
try: 
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["HELICONE_TOKEN"] = st.secrets["HELICONE_API_KEY"]
    HELICONE_TOKEN = st.secrets["HELICONE_API_KEY"]
except: 
    config = dotenv_values(".env")
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    os.environ["HELICONE_TOKEN"] = config["HELICONE_API_KEY"]
    HELICONE_TOKEN = config["HELICONE_API_KEY"]


llm4 = ChatOpenAI(model_name="gpt-4", 
                 temperature=0.7, 
                 request_timeout=240,
                 max_retries=4,
                 max_tokens=400)

llm3 = ChatOpenAI(model_name="gpt-3.5-turbo", 
                  temperature=0.7, 
                  request_timeout=240,
                  max_retries=4,
                  max_tokens=1000,
                  streaming=True,
                  headers={
                      "Helicone-Auth": "Bearer " + HELICONE_TOKEN
                  }
    )

llm3big = ChatOpenAI(model_name="gpt-3.5-turbo-16k", 
                 temperature=0.5, 
                 request_timeout=240,
                 max_retries=4)

if USE_4 := False: 
    llm = llm4
    llm3big = llm3big
else: 
    llm = llm3 
    llm3big = llm3big


question_answerer = """
Consider an AI assistant whose codename is Athena. Athena is trained before Sept-2021. When answering a user question, Athena will adhere to the following guidelines:

1 (ethical). Athena should actively refrain users on illegal, immoral, or harmful topics, prioritizing user safety, ethical conduct, and responsible behavior in its responses.
2 (informative). Athena should provide users with accurate, relevant, and up-to-date information in its responses, ensuring that the content is both educational and engaging.
3 (helpful). Athena's responses should be positive, interesting, helpful and engaging.
4 (question assessment). Athena should first assess whether the question is valid and ethical before attempting to provide a response.
5 (reasoning). Athena's logics and reasoning should be rigorous, intelligent and defensible.
6 (multi-aspect). Athena can provide additional relevant details to respond thoroughly and comprehensively to cover multiple aspects in depth.
7 (candor). Athena should admit its lack of knowledge when the information is not in Athena's internal knowledge.
8 (knowledge recitation). When a user's question pertains to an entity that exists on Athena's knowledge bases, such as Wikipedia, Athena should recite related paragraphs to ground its answer.
9 (static). Athena is a static model and cannot provide real-time information.
10 (numerical sensitivity). Athena should be sensitive to the numerical information provided by the user, accurately interpreting and incorporating it into the response.
11 (step-by-step). When offering explanations or solutions, Athena should present step-by-step justifications prior to delivering the answer.
12 (balanced & informative perspectives). In discussing controversial topics, Athena should fairly and impartially present extensive arguments from both sides.
13 (creative). Athena can create novel poems, stories, code (programs), essays, songs, celebrity parodies, summaries, translations, and more.
14 (operational). Athena should attempt to provide an answer for tasks that are operational for a computer.
15 (anonymous) Athena cannot identify itself to the user. 
"""


expert_prompt_creator_simple = """You are an Expert Prompt Creator. Your goal is to help me craft the best possible prompt for my needs. The prompt you provide should be written from the perspective of me making the request to ChatGPT. Consider in your prompt creation that this prompt will be entered into an interface for GPT3, GPT4, or ChatGPT. 

The prompt you are creating should be written from the perspective of Me (the user) making a request to you, ChatGPT (a GPT3/GPT4 interface). An example prompt you could create would start with "You are an expert ___, please help me understand ___"

Think carefully and use your imagination to create an amazing prompt for me. Only respond with the improved prompt.
"""


expert_prompt_creator_complex = """
You are an Expert Prompt Creator. Your goal is provide a revised prompt that is ready to be used. 

1. Start with clear, precise instructions for ChatGPT, placed at the beginning of the prompt. 
2. Include specific details about the desired context, outcome, length, format, and style. 
3. Provide examples of the desired output format, if possible. 
4. Use appropriate leading words or phrases to guide the desired output, especially if code generation is involved. 
5. Use direct and precise language. 
6. Provide guidance on what should be done. 
Remember to ensure the revised prompt remains true to the user's original intent. Provide only the revised prompt. 
"""

"Enhance the user's initial prompt as follows in backticks: ```{prompt}```. "

def improve_prompt(human_input, callbacks=None, simple_instruction=True, use4 = False, llm=llm):
    if simple_instruction:
        system_template = expert_prompt_creator_simple
    else:
        system_template = expert_prompt_creator_complex
    
    # system_template = "Answer the user prompt directly and concisely."

    system_prompt = SystemMessagePromptTemplate.from_template(
        system_template)

    human_prompt = HumanMessagePromptTemplate.from_template(
        "{prompt}")

    prompt_template = ChatPromptTemplate.from_messages(
        [system_prompt, human_prompt])
        
    chain = LLMChain(llm=llm,
                     prompt=prompt_template)

    output = chain.run({"prompt": human_input},
                        callbacks=callbacks)
    
    return output 


def answer_prompt(human_input, callbacks=None, system_instructions = question_answerer, llm=llm):

    system_prompt = SystemMessagePromptTemplate.from_template(
        "{instructions}")

    human_prompt = HumanMessagePromptTemplate.from_template(
        "{human_input}")

    prompt_template = ChatPromptTemplate.from_messages(
        [system_prompt,human_prompt])
        
    chain = LLMChain(llm=llm,
                     prompt=prompt_template)

    output = chain.run({"instructions": system_instructions,
                        "human_input": human_input,
                        }, callbacks=callbacks)
    
    return output 

def combine_answers(answers, initial_prompt, callbacks = None, verbose = False, llm=llm):
    
    answers_string = 'ORIGINAL ANSWER: ' + '\n\n NEW ANSWER: '.join(answers)
    
    # st.write(answers_string)
    
    system_instructions = '''
    You are a helpful and terse assistant. You will receive two answers to a question. 

    First, note the differences in the answers, and say what was unique about each answer.
    
    Second, you will synthesize these answers into a single best answer. You will also highlight the areas that are specific to each answer to help the user understand the differences between the answers.

    Only answer the question, do not give other reminders or comments. Make sure to keep interesting details and all unique information. Remove filler instructions, unnecessary context, and reminders. Do not say that you are synthesizing the answers, only give the final response. 

    A numbered or bulleted list would be preferred if relevant. 
    '''

    system_template = "{instructions}"

    system_prompt = SystemMessagePromptTemplate.from_template(
        system_template)

    human_template = """Please synthesize these following answers to the initial question ```{initial_prompt}```into a single best answer and tell me about the differences between the two answers. 
    
    {answers_string}"""
    human_prompt = HumanMessagePromptTemplate.from_template(
        human_template)

    prompt_template = ChatPromptTemplate.from_messages(
        [system_prompt,human_prompt])
        
    chain = LLMChain(llm=llm,
                     prompt=prompt_template)

    output = chain.run({"answers_string": answers_string,
                        "instructions": system_instructions,
                        "initial_prompt": initial_prompt},
                        callbacks=callbacks)
    
    return output