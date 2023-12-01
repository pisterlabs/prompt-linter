import logging
import json
import pandas as pd
import os
import nltk
import spacy
import streamlit as st
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime as dt

from langchain.llms import OpenAI, Cohere
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank, LLMChainExtractor
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.schema import OutputParserException
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    HumanMessagePromptTemplate, ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate
)

# REMEMBER TO ADD YOUR API KEYS HERE
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
os.environ["WOLFRAM_ALPHA_APPID"] = st.secrets["WOLFRAM_ALPHA_APPID"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
# ----------

# TODO: Fix build_chain function
# TODO: Write generic function to build custom langchain tools (i.e., summarise, suggest, search-chat)
# TODO: Write functions to save and load information from memory
# TODO: Implement asynchronous versions of llm/chain builders
# TODO: Rewrite build_llm to support multiple providers, and their respective models

#model = "google/flan-t5-base"
    #llm = HuggingFaceHub(repo_id=model,
    #                 model_kwargs={"temperature": 0.9,
    #                               "max_length": 100})

# interestingly the es_core_news_sm dictionary in spanish is better at identifying entities than the english one
# python -m spacy download en_core_web_sm <- run in terminal to download the english dictionary (es_core_news_sm for spanish)
#nlp = spacy.load("en_core_web_sm")

@st.cache_data(max_entries=10, ttl=3600, show_spinner=True)
def download_cache_report(report):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return report

def compute_cost(tokens, engine):
    
    model_prices = {"text-davinci-003": 0.02, 
                    "gpt-3.5-turbo": 0.002, 
                    "gpt-4": 0.03,
                    "cohere-free": 0}
    model_price = model_prices[engine]
    
    cost = (tokens / 1000) * model_price

    return cost

def build_llm(max_tokens: int, 
              temperature: int, 
              provider: str):
    '''
    Function to build a LLM model using lanchain library. 
    Default model is text-davinci-003 for OpenAI provider, but you can change it to any other model depending on the provider's models.
    note that for chat models you would set provider = "ChatOpenAI" for example.
    params:
        max_tokens: int, default 260
        temperature: float, default 0.6
        provider: str, default 'openai'
    return:
        llm: Langchain llm object
    '''
    llm = None
    
    if provider == "openai":
        llm = OpenAI(model_name='text-davinci-003', temperature=temperature, max_tokens=max_tokens)
        
    elif provider == "ChatGPT3":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=max_tokens)
    
    elif provider == "ChatGPT4":
        llm = ChatOpenAI(model_name="gpt-4", temperature=temperature, max_tokens=max_tokens)
    
    elif provider == "cohere":
        llm = Cohere(temperature=temperature, max_tokens=max_tokens)
    
    return llm

def read_perma4(path: str, return_test_answers=False):
    '''
    Function to read perma4 dataset.
    params:
        path: str to json file
        return_test_answers: bool, default False
    return:
        data: pandas dataframe
    '''
    data = pd.read_json(path)
    
    questions = data['questions']
    
    if return_test_answers:
        return data
    else:
        return questions
    
def memory_to_pandas(memory_path: str):
    '''
    Function to convert memory to pandas dataframe.
    params:
        memory_path: path to memory json file
    return:
        df: pandas dataframe
    '''
    with open(memory_path) as f:
        data = json.load(f)    
    
    return data

def build_report(report_generator_profile: dict,
                 perma_results: pd.DataFrame,
                 user_data: list):
    '''
    Function to initialize and run report generator given the AI profile and perma4 results.
    params:
        report_generator_profile: dict
        perma_results: list
    return:
        report: str
    '''
    questions = perma_results['Questions:']
    demo_answers = perma_results['Answers:']
    
    name = report_generator_profile['name']
    agent_type = report_generator_profile['agent_type']
    personality = report_generator_profile['personality']
    knowledge = report_generator_profile['knowledge']
    tools = report_generator_profile['tools']
    keywords = report_generator_profile['keywords']
    description = report_generator_profile['description']
    max_tokens = report_generator_profile['max_tokens']
    temperature = report_generator_profile['temperature']
    
    report_structure = "1. Positive Emotions \n 2. Engagement \n 3. Relationships \n 4. Meaning \n 5. Accomplishment \n 6. Physical Health \n 7. Mindset \n 8. Work Environment \n 9. Economic Security"
 
    sys_prompt_template = '''You are {name}, an expert in [{knowledge}] with {personality} personality. {description}. You can use the following keywords to help you: {keywords} '''
    task_prompt_template = '''Use the following questions {questions} and responses {demo_answers} to provide a well being assessment of the surveyed object with the following properties {user_data} based on the 9 pillars of Perma+4 framework.
    The output must be a structured, insightful and concise report that associates the responses to the questions with the 9 pillars of Perma+4 framework. 
    Here is an example of the desired structure {report_structure}. 
    
    --{user_name}'s REPORT-- 
    
    '''
    prompt_template = sys_prompt_template + task_prompt_template
    
    prompt = PromptTemplate(input_variables=[
                            "name", "knowledge", "description", "keywords", "user_name", "user_data",
                            "questions", "demo_answers", "personality", "report_structure"
                            ],
                            template=prompt_template)
    
    # default of build_llm is text-davinci-003
    engine = build_llm(max_tokens=max_tokens, temperature=temperature)
    
    chain = LLMChain(llm=engine, prompt=prompt)
    report = chain.run({'name': name,
                          'knowledge': knowledge,
                          'description': description,
                          'user_data':user_data,
                          'keywords': keywords,
                          'user_name':user_data[0],
                          'questions': questions,
                          'demo_answers': demo_answers,
                          'personality': personality,
                          'report_structure': report_structure})
    
    # get number of tokens in report
    tokens = len(report.split())
    # cost of report
    report_cost = compute_cost(tokens, 'text-davinci-003')
    
    return report, report_cost

def create_qa_pairs(dataframe):
    qa_pairs = ""
    
    for _, row in dataframe.iterrows():
        question = row['Question']
        answer = row['Response']
        qa_pairs += f"{question}:{answer}; "
    
    return qa_pairs.rstrip("; ")
    

def build_pillar_report(report_generator_profile: dict,
                        pillar: str,
                        perma_results: pd.DataFrame,
                        user_data: list,
                        provider: str):
    '''
    Function to initialize and run report generator given the AI profile and perma4 results for a specific pillar (workshop).
    params:
        report_generator_profile: dict
        perma_results: list
    return:
        report: str
    '''
    #questions = perma_results['Question']
    #st.write(questions)
    #st.write(type(questions))
    #demo_answers = perma_results['Response']
    
    name = report_generator_profile['name']
    agent_type = report_generator_profile['agent_type']
    personality = report_generator_profile['personality']
    description = report_generator_profile['description']
    max_tokens = report_generator_profile['max_tokens']
    temperature = report_generator_profile['temperature']
    
    #report_structure = "1. Positive Emotions \n 2. Engagement \n 3. Relationships \n 4. Meaning \n 5. Accomplishment \n 6. Physical Health \n 7. Mindset \n 8. Work Environment \n 9. Economic Security"
 
    sys_prompt_template = '''You are {description} with a {personality} personality.'''
    
    ## string formatting to parse series of questions and answers and create question-answer pairs
    qa_pairs = create_qa_pairs(perma_results)
    st.write(qa_pairs)

    task_prompt_template = '''
    Given the following question-answer pairs ({qa_pairs}), provide a comprehensive well-being assessment of the individual with attributes: {user_data}. Your analysis should focus on the {pillar} pillar of the Perma+4 framework. 

    Present your findings in a concise report format. Begin with a user profile overview for well-being coaches. Follow with a detailed analysis, connecting the responses to the Perma+4 {pillar} pillar questions.

    -- {user_name}'s Well-Being Assessment Report -
    '''

    
    # default of build_llm is text-davinci-003
    if provider == "openai":
        engine = build_llm(max_tokens=max_tokens, temperature=temperature, 
                       provider="openai")
        
        prompt_template = sys_prompt_template + task_prompt_template
    
        prompt = PromptTemplate(input_variables=[
                                "description", "user_name", "user_data", "qa_pairs", "personality", "pillar"
                                ],
                                template=prompt_template)
        
        chain = LLMChain(llm=engine, prompt=prompt)
        
        report = chain.run({'description': description,
                            'user_data':user_data,  
                            'user_name':user_data[0],
                            'qa_pairs': qa_pairs,
                            'personality': personality,
                            'pillar': pillar})
        
        # get number of tokens in report
        tokens = len(report.split())
        # cost of report
        report_cost = compute_cost(tokens, 'text-davinci-003')
        
    elif provider == "ChatGPT4":
        engine = build_llm(max_tokens=max_tokens, temperature=temperature,
                          provider="ChatGPT4")
    
        messages = [SystemMessage(content=sys_prompt_template),
                HumanMessage(content=task_prompt_template)]
        
        # run chat llm
        report = engine(messages).content
        
        # cost of report
        report_cost = compute_cost(len(report.split()), 'gpt-4')
    
    return report, report_cost


# TODO: Add curated data source to complement google search results
def suggest_activities(coach_profile: dict,
                       report: str,
                       provider: str):
    '''
    Function to initialize and run coach given the AI profile and user data.
    params:
        coach_profile: dict
        user_data: list
    return:
        activities: str
    '''
    name = coach_profile['name']
    personality = coach_profile['personality']
    knowledge = coach_profile['knowledge']
    description = coach_profile['description']
    max_tokens = coach_profile['max_tokens']
    temperature = coach_profile['temperature']
    
    sys_prompt_template = "You are {name}, an expert specializing in {knowledge} and embodying a {personality} disposition. With a deep understanding of evidence-backed strategies, your aim is to provide tailored recommendations that enhance individuals' wellbeing. {description}"

    task_prompt_template = """Use the following insights from the provided wellbeing report to generate three practical and achievable activities that will effectively elevate the individual's wellbeing. For each goal, provide a brief explanation of why it's recommended.
    --
    Wellbeing Report: {report}
    --

    Examples:
    1. Activity: Incorporate mindfulness meditation for 10 minutes daily.\n
    Explanation: Mindfulness meditation has been shown to reduce stress, enhance self-awareness, and promote overall mental clarity.

    2. Activity: Engage in regular physical activity, such as brisk walking, for at least 30 minutes every day.\n
    Explanation: Physical activity releases endorphins, which can improve mood and energy levels while supporting cardiovascular health.

    3. Activity: Maintain a gratitude journal to jot down three things you're thankful for each day.\n
    Explanation: Practicing gratitude fosters a positive mindset and has been linked to increased life satisfaction and reduced negative emotions.
    --
    Remember to structure your responses clearly and provide actionable insights for each recommendation:
    
    """
    
    #search = GoogleSerperAPIWrapper()
    #tools = [(Tool(name='Intermediate Answer',
    #              func=search.run,
     #             description="useful for when you need to ask with search"))]
    
    prompt_template = sys_prompt_template + task_prompt_template
    
    engine = build_llm(max_tokens=max_tokens, temperature=temperature, 
                       provider=provider)
    
    #coach_agent = initialize_agent(tools,
    #                              engine,
    #                               agent=AgentType.SELF_ASK_WITH_SEARCH,
    #                               verbose=True, max_iterations=10, early_stopping_method="generate"
    #                               )
    
    prompt = PromptTemplate(input_variables=[
                            "description", "name", "knowledge", "personality", "report",
                            ],
                            template=prompt_template)
    
    chain = LLMChain(llm=engine, prompt=prompt)
    
    activities = chain.run({'description': description,
                        'name': name,
                        'knowledge': knowledge,
                        'personality': personality,
                        'report': report})
    

    tokens_used = len(activities.split())
    activities_cost = compute_cost(tokens_used, 'text-davinci-003')
        
    return activities, activities_cost


## 08/17/2023
def chat_smart_goal(smart_profile: dict,
                   report: str,
                   human_input: str):
    '''
    Function to set smart goal conversationally with LLM.
    params:
        smart_profile: dict
        provider: str
    return:
        smart_goal: str
    '''
    #name = smart_profile['name']
    persona = smart_profile['system_prompt']
    temperature = smart_profile['temperature']
    max_tokens = smart_profile['max_tokens']
    

    system_template = SystemMessagePromptTemplate.from_template(
                    """You are a thoughtful, analytical, and empathetic wellness coach. 
                    You specialize in helping clients turn vague goals into SMART goals that are Specific, Measurable, Achievable, Relevant, and Time-Bound. \n
                    You have the following information about the client you are currently working with {report}."""
                    )
    human_template = HumanMessagePromptTemplate.from_template("{input}")
    ai_template = AIMessagePromptTemplate.from_template("{response}")

    # create the list of messages
    chat_prompt = ChatPromptTemplate.from_messages([
        system_template,
        human_template,
        ai_template
    ])
    
    # build chat llm
    chatgpt = build_llm(provider='ChatGPT4', 
                        max_tokens=max_tokens, temperature=temperature)
    
    # Build chain
    conversation_chain = ConversationChain(llm=chatgpt, prompt=chat_prompt,
                                           memory=ConversationBufferMemory())
    
    # run chat llm
    llm_output = conversation_chain.run({'report': report,
                                        'input': human_input,
                                        'response': "Hi! I'm your wellness coach. I'm here to help you set a SMART goal. Shall we defining the first dimension?"})
    
    # cost of report
    llm_cost = compute_cost(len(llm_output.split()), 'gpt-4')
    
    return llm_output, llm_cost

def completion_smart_goal(smart_profile: dict,
                          report: str,
                          user_goal: str,
                          provider: str):
    '''
    One-shot completion of smart goal.
    '''
    smart_goal = ""
    
    persona = smart_profile['system_prompt']
    temperature = smart_profile['temperature']
    max_tokens = smart_profile['max_tokens']
    
    # SMART report structure
    output_structure = "1. Specific:\n2. Measurable:\n3. Achievable:\n4. Relevant:\n5. Time-bound:\n"

    context = "Your expertise lies in helping clients create effective SMART goals. You are currently working with a new client who has shared their information with you. Here's the context of the client's situation:\n\n{report}\n"
    sys_prompt_template = persona + context

    # Task prompt to build a detailed and formatted SMART goal
    task_prompt_template = '''The user has set the following goal: "{user_goal}".\n
    Please help the user formulate a SMART goal. Break down the user's goal into actionable steps and enumerate each component clearly based on the following structure:\n{output_structure}
    Then lay out the entire SMART goal for me in a single paragraph. Proceed step by step. Provide a clear and concise plan. Do not make reference to any particular year.\n
    --
    '''
    
    if provider == 'openai':
        # prompt template   
        prompt_template = sys_prompt_template + task_prompt_template
        prompt = PromptTemplate(input_variables=["report", "user_goal", "output_structure"],
                                template=prompt_template)
        
        # build llm
        davinci = build_llm(max_tokens=max_tokens, temperature=temperature, 
                            provider='openai')
        
        # build chain
        chain = LLMChain(llm=davinci, prompt=prompt)
        
        smart_goal = chain.run({'report': report,
                                'user_goal': user_goal,
                                'output_structure': output_structure})
        
        # cost of report
        smart_goal_cost = compute_cost(len(smart_goal.split()), 'text-davinci-003')
        
    elif provider == 'ChatGPT4':
        
        engine = build_llm(max_tokens=max_tokens, temperature=temperature,
                           provider='ChatGPT4')
        
        messages = [SystemMessage(content=sys_prompt_template),
                HumanMessage(content=task_prompt_template)]
        
        # run chat llm
        smart_goal = engine(messages).content
        
        # cost of report
        smart_goal_cost = compute_cost(len(smart_goal.split()), 'gpt-4')
    
    return smart_goal, smart_goal_cost
    
def completion_obstacles_and_planning(goal: str, smart_goal: str):
    '''
    Function to identify and plan for potential internal or external obstacles of a given SMART goal. 
    Functionality based on the Obstacles and Planning components of the WOOP framework.
    
    params:
        smart_goal: str
    return:
        obstacles: str
    '''
    
    sys_prompt = "You are a professional wellness coach who has received their certification from the International Coaching Federation."\
                + " You specialize in helping users identify potential obstacles from their SMART goal, and set a plan to address such obstacles, based on the WOOP framework." \
                + " Address the user as if you were their wellbeing coach. \n"
    
    task_prompt = "After stating the goal to work on ({goal}) write a list of potential internal and external obstacles that may prevent the user from achieving their SMART goal: {smart_goal}. \n -- \n Then, recommend a plan to help the user think about how to overcome each obstacle. Proceed step by step. \n\n"
    
    prompt = PromptTemplate(input_variables=["goal", "smart_goal"],
                            template=sys_prompt + task_prompt)
    
    davinci = build_llm(max_tokens=350, temperature=0.85, 
                       provider='openai')
    
    chain = LLMChain(llm=davinci, prompt=prompt)
    
    obstacles = chain.run({'smart_goal': smart_goal,
                           'goal': goal})
    
    # cost
    cost = compute_cost(len(obstacles.split()), 'text-davinci-003')
    
    return obstacles, cost

def send_email(user_name, user_email, feedback):
    smtp_port = 587                 # Standard secure SMTP port
    smtp_server = "smtp.gmail.com"  # Google SMTP Server

    email_from = "agbonorino21@gmail.com" 
    email_to = "atlas.intelligence21@gmail.com" 

    pswd = "arqxfjbvwuyybopi"       # App password for gmail
    
    message = f"""Feedback from {user_name} ({user_email}):

    {feedback}
    """

    ### SEND EMAIL ###
    simple_email_context = ssl.create_default_context()

    try:
        # Connect to the server
        print("Connecting to server...")
        TIE_server = smtplib.SMTP(smtp_server, smtp_port)
        TIE_server.starttls(context=simple_email_context)
        TIE_server.login(email_from, pswd)
        print("Connected to server :-)")
        
        # Create the MIMEText object with the message and encoding
        msg = MIMEMultipart()
        msg.attach(MIMEText(message, "plain", "utf-8"))
        msg["From"] = email_from
        msg["To"] = email_to
        msg["Subject"] = "Feedback Form - Atlas Demo ({})".format(user_name)
        
        # Send the actual email
        print()
        print(f"Sending email to - {email_to}")
        TIE_server.send_message(msg)
        print(f"Email successfully sent to - {email_to}")

    # If there's an error, print it out
    except Exception as e:
        print(e)

    # Close the port
    finally:
        TIE_server.quit()