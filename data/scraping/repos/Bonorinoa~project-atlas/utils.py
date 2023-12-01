import logging
import json
import pandas as pd
import os
import nltk
import spacy

from langchain import OpenAI
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

# REMEMBER TO ADD YOUR API KEYS HERE
# os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPER_API_KEY"] = "ef63b458e83c92aea9903b0b6ee7aae63872b5a7"
os.environ["WOLFRAM_ALPHA_APPID"] = "42V6VG-3TEWJ62W3Y"
# ----------

# TODO: Fix build_chain function
# TODO: Write generic function to build custom langchain tools (i.e., summarise, suggest, search-chat)
# TODO: Write functions to save and load information from memory
# TODO: Implement asynchronous versions of llm/chain builders

# interestingly the es_core_news_sm dictionary in spanish is better at identifying entities than the english one
# python -m spacy download en_core_web_sm <- run in terminal to download the english dictionary (es_core_news_sm for spanish)
nlp = spacy.load("en_core_web_sm")

# entities and keywords from query  
def extract_entities_keywords(text):
    '''
    Function to extract entities and keywords from a text using spacy library.
    params:
        text: str
    return:
        entities: list
        keywords: list    
    '''
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    keywords = [token.lemma_ for token in doc if token.is_stop == False and token.is_punct == False]
    return entities, keywords

def compute_cost(tokens, engine):
    
    model_prices = {"text-davinci-003": 0.02, 
                    "gpt-3.5-turbo": 0.002, 
                    "gpt-4": 0.03}
    model_price = model_prices[engine]
    
    cost = (tokens / 1000) * model_price

    return cost

def build_coach_llm(max_tokens=560,
                    temperature=0.7,
                    provider="openai"):
    
    '''
    Function to build main LLM personality model for Atlas Coach using Langchain.
    This LLM is meant to take in the user query, determine which agent(s) to use, wait for the agent(s) to return a response, and then use the response as context to answer the query.
    params:
        max_tokens: int, default 560
        temperature: float, default 0.7
        provider: str, default 'openai'
    return:
        llm: Langchain llm object
    '''
    gpt3_coach = OpenAI(model_name='text-davinci-003', 
                        temperature=temperature, max_tokens=max_tokens)
    
    prompt_template_coach = '''{something} and something else'''

def build_llm(max_tokens: int, 
              temperature: int, 
              provider="openai"):
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
    elif provider == "ChatOpenAI":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=max_tokens)
    
    return llm

def build_llm_tools(tools: list,
                    max_tokens=260, 
                    temperature=0.6, 
                    provider="openai"):
    '''
    Function to build agent (llm + tools) using lanchain library.
    params:
        tools: list of tools
        model_name: str, default 'text-davinci-003'
        max_tokens: int, default 260
        temperature: float, default 0.6
        provider: str, default 'openai'
    return:
        agent: Langchain agent object
    '''
    agent = None
    if provider == "openai":
        llm = build_llm(temperature=temperature, max_tokens=max_tokens, provider=provider)

        tools = load_tools(tools, llm=llm)

        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        
    return agent

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
                 perma_results: list):
    '''
    Function to initialize and run report generator given the AI profile and perma4 results.
    params:
        report_generator_profile: dict
        perma_results: list
    return:
        report: str
    '''
    questions = perma_results['questions']
    demo_answers = perma_results['demo_answers']
    
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
    task_prompt_template = '''Use the following questions {questions} and responses {demo_answers} Provide a well being assessment of the surveyed object based on the 9 pillars of Perma+4 framework.
    The output must be a structured and conciser report that associates the responses to the questions with the 9 pillars of Perma+4 framework. 
    Here is an example of the desired structure {report_structure}. REPORT: 
    
    1. Positive Emotions:
    '''
    prompt_template = sys_prompt_template + task_prompt_template
    
    prompt = PromptTemplate(input_variables=[
                            "name", "knowledge", "description", "keywords", 
                            "questions", "demo_answers", "personality", "report_structure"
                            ],
                            template=prompt_template)
    
    # default of build_llm is text-davinci-003
    engine = build_llm(max_tokens=max_tokens, temperature=temperature)
    
    chain = LLMChain(llm=engine, prompt=prompt)
    report = chain.run({'name': name,
                          'knowledge': knowledge,
                          'description': description,
                          'keywords': keywords,
                          'questions': questions,
                          'demo_answers': demo_answers,
                          'personality': personality,
                          'report_structure': report_structure})
    
    return report
    
def generate_goals(recommender_generator_profile: dict,
                   user_data: list,
                   report: str):
    '''
    Function to initialize and run recommender generator given the AI profile and user data.
    params:
        recommender_generator_profile: dict
        user_data: list
    return:
        goals: str
    '''
    name = recommender_generator_profile['name']
    agent_type = recommender_generator_profile['agent_type']
    personality = recommender_generator_profile['personality']
    knowledge = recommender_generator_profile['knowledge']
    tools = recommender_generator_profile['tools']
    keywords = recommender_generator_profile['keywords']
    description = recommender_generator_profile['description']
    max_tokens = recommender_generator_profile['max_tokens']
    temperature = recommender_generator_profile['temperature']
    
    sys_prompt_template = '''You are {name}, an expert in [{knowledge}] with {personality} personality. {description}.'''
    task_prompt_template = '''Use the following user data {user_data} and insights {report} to provide three suggested goals 
    for the surveyed object that will maximize his net benefit for the effort required to improve along the dimensions that need the most improvement. 
    GOALS: 
    
    '''
    
    prompt_template = sys_prompt_template + task_prompt_template
    
    prompt = PromptTemplate(input_variables=["name", "knowledge", "description", 
                                             "user_data", "report", "personality"],
                            template=prompt_template)
    
    engine = build_llm(max_tokens=max_tokens, temperature=temperature)
    
    chain = LLMChain(llm=engine, prompt=prompt)
    
    goals = chain.run({'name': name,
                    'knowledge': knowledge,
                    'description': description,
                    'user_data': user_data,
                    'report': report,
                    "personality": personality})
    
    return goals

def suggest_activities(coach_profile: dict,
                       user_data: list,
                       goals: str):
    '''
    Function to initialize and run coach given the AI profile and user data.
    params:
        coach_profile: dict
        user_data: list
    return:
        activities: str
    '''
    name = coach_profile['name']
    agent_type = coach_profile['agent_type']
    personality = coach_profile['personality']
    knowledge = coach_profile['knowledge']
    tools = coach_profile['tools']
    keywords = coach_profile['keywords']
    description = coach_profile['description']
    max_tokens = coach_profile['max_tokens']
    temperature = coach_profile['temperature']
    
    sys_prompt_template = f'''You are {name}, an expert in [{knowledge}] with a {personality} personality. {description}.'''
    task_prompt_template = f'''Given the following user data {user_data} and suggested goals {goals}.
    Recommend two activities per goal that will help the surveyed object achieve or move towards the suggested goals.
    
    Prioritize the tools as follows: 
    1. google-serper to research relevant wellbeing activities that can be recommended to this user. 
    You can use the following keywords to optimize your search: {keywords}.
    
    Write the suggested activities in bullet point format for each goal in a clear and structured format. ACTIVITIES: 
      
    '''
    
    #search = GoogleSerperAPIWrapper()
    #tools = [(Tool(name='Intermediate Answer',
    #              func=search.run,
    #              description="useful for when you need to ask with search"))]
    
    tools = load_tools(['google-serper', 'wolfram-alpha'])
    
    prompt_template = sys_prompt_template + task_prompt_template
    
    engine = build_llm(max_tokens=max_tokens, temperature=temperature)
    coach_agent = initialize_agent(tools,
                                   engine,
                                   agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                   verbose=True)
    try:
        activities = coach_agent.run(prompt_template)
    except OutputParserException as e:
        activities='1.'+str(e).split('1.')[1]
    return activities

# ignore this function, is still on development. Not sure if it will be useful or not.
def run_agent_from_profile(agent_profile: dict, 
                           query: str):
    '''
    Function to build agent from memory using lanchain library.
    params:
        agent_profile: dict
        memory: pandas dataframe
    return:
        agent: Langchain agent object
    '''
    agent = None 
    
    name = agent_profile['name']
    agent_type = agent_profile['agent_type']
    personality = agent_profile['personality']
    knowledge = agent_profile['knowledge']
    tools = agent_profile['tools']
    description = agent_profile['description']
    max_tokens = agent_profile['max_tokens']
    temperature = agent_profile['temperature']
    
    engine = build_llm(model_name='text-davinci-003', 
                       max_tokens=max_tokens, temperature=temperature)
    llm_tools = load_tools(tools, llm=engine)
    
    prompt_template = '''You are {name}. {description}. You have a {personality} personality and {knowledge} knowledge.'''
    prompt = PromptTemplate(input_variables=[name, description, query, personality, knowledge],
                            template=prompt_template)
    
    if agent_type == "zeroShot":
        print(f"Building (zeroShot) {name} agent...")
        zeroShot_agent = initialize_agent(tools=llm_tools, llm=engine, 
                                 agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        
        # Build prompt before running to specify custom agent's prompt using its description, personality, and knowledge.
        ## 
        
        zeroShot_chain = LLMChain(llm=engine,
                                  prompt=prompt)
                                  
        agent_response = zeroShot_chain.run(query)
        
        
        #agent = zeroShot_agent
    
    elif agent_type == "selfAskSearch":
        print(f"Building (selfAskSearch) {name} agent...")
        search = GoogleSerperAPIWrapper()
        # intermediate answer tool
        self_tools = [Tool(name="Intermediate Answer",
                        func=search.run,
                        description="useful for when you need to ask with search")]
        
        sealfAsk_agent = initialize_agent(tools=self_tools, llm=engine, 
                                 agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
        agent = sealfAsk_agent
    
    return agent_response

