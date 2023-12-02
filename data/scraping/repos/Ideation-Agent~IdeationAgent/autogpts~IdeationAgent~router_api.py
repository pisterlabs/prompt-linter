import os
from typing import Any

import openai
import json
import requests
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import initialize_agent, Tool
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import AgentType
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from selenium import webdriver
from linkedin_scraper import Person, actions
from selenium.webdriver.chrome.options import Options
from langchain.schema import SystemMessage
from serpapi import GoogleSearch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class LinkedinURL(BaseModel):
    linkedin_url: str

class IdeaSource(BaseModel):
    linkedin_summary: str
    interests: str = None

class DiscussionSource(BaseModel):
    log: dict[str, Any]
    human_input: str
    speaker_list: list[int]

class BusinessPlanSource(BaseModel):
    log: dict[str, Any]

load_dotenv('.env')
serper_api_key = os.getenv('SERP_API_KEY')
browserless_api_key = os.getenv('BROWSERLESS_API_KEY')
linkedin_email = os.getenv('LINKEDIN_EMAIL')
linkedin_password = os.getenv('LINKEDIN_PASSWORD')
openai.api_key = os.getenv('OPENAI_API_KEY')

def search(query):
    params = {
    "engine": "google",
    "q": query,
    "api_key": serper_api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    return str(organic_results)

def scrape_website(url):
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    data = {"url": url}
    data_json = json.dumps(data)
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

        objective = "summarize"

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

def summary(content, objective):
    # The agent processes the content and generates a concise summary.
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    # Tool(
    #     name="ScrapeWebsite",
    #     func=scrape_website,
    #     description="Scrape content from a website"
    # )
]

llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k-0613')
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

@app.get("/")
async def root():
	return { "message" : "Hello World" }

@app.post("/check_bio/")
async def check_bio(linkedin_url: LinkedinURL):
    print(f"linkedin_url: , {linkedin_url.linkedin_url}")
    options = Options()
    options.add_argument('--headless=new')
    driver = webdriver.Chrome(options=options)
    actions.login(driver, linkedin_email, linkedin_password)
    person = Person(linkedin_url.linkedin_url, driver=driver)
    summary = person.about + str(person.experiences) + str(person.educations) + str(person.interests) + str(person.accomplishments)
    return {"profile_name": person.name, "linkedin_summary": summary}

@app.post("/generate_initial_ideas/")
async def generate_initial_ideas(idea_source: IdeaSource):
    user_input = f"linkedin_summary : , {idea_source.linkedin_summary}, interests : , {idea_source.interests}"
    print(user_input)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", 
                    "content": """
                    You are a world class business idea generator, who can generate the best startup ideas. 
                    If the user provide linkedin information and interests, you will try to generate the 5 best startup ideas.
                    You should generate service name, define clear problem and solution for each idea.
                    Reply only in json with the following format:

                    {
                        \"ideas\": {
                            \"service_name\":  \"name of service\",
                            \"problem\": \"original and clear problem definition, not phenomenon\",
                            \"service_idea\": \"service idea should address the problem clearly\",
                        },
                    }

                    """},
                    {"role": "user", "content": user_input},
                ]
            )
            ideas_str_str = response["choices"][0]["message"]["content"]
            ideas = json.loads(ideas_str_str)
            break
        except:
            continue
    return ideas

speaker_list = []
@app.post("/discuss/")
async def discuss(ds: DiscussionSource):
    print(f"log :, {ds.log}, human_input:, {ds.human_input}, speaker_list:, {ds.speaker_list}")

    global speaker_list
    dialog = [{"CEO": "Ok. let's start discussion."}]
    speaker_names = {
        0: "Pat",
        1: "Lily",
        2: "Steve",
        3: "Casey",
        4: "CEO"
    }

    if len(ds.log["dialog"]) == 0:
        speaker_list = []
        speaker_list.extend([0, 1, 2, 3])
    else:
        dialog.extend(ds.log["dialog"])

    speaker_list.extend(ds.speaker_list)
    if len(speaker_list) != 0:
        speaker_num = int(speaker_list[0])
        del speaker_list[0]
    else: # Start AutoGPT
        url = "http://localhost:8000/ap/v1/agent/tasks/"
        input_message = """
        You are a world-class CEO of a startup.
        You should run the discussion in a fact based manner to generate realistic detailed business plan.

        Reply only in one json with the following format:

        {
            \"member number (repliy only in integer from 0 to 3)\": \"give feedback to the idea or ask to elaborate on the idea\"
        }

        Here is the discussion so far:

        """
        input_message += str(dialog)
        input_message += f"Here is the comment of cheif of staff: {ds.human_input}"
        print(input_message)
        input = json.dumps({"input": input_message, "additional_input": ""})
        res = requests.post(url, data=input)
        res = json.loads(res.text)
        task_id = res["task_id"]
        url = f"http://localhost:8000/ap/v1/agent/tasks/{task_id}/steps/"
        input = json.dumps({"task_id": task_id, "step": {"input":input_message, "additional_input": ""}})
        while True:
            try:
                res = requests.post(url, data=input)
                res = json.loads(res.text)
                ceo_feedback = res["output"]
                speaker_list.extend(ceo_feedback.keys())
                new_answer = {}
                for k in ceo_feedback:
                    v = ceo_feedback[k]
                    new_answer[speaker_names[int(k)]] = v
                
                response = {"speaker": 4, "contents": str(new_answer), "is_finished": False}
                break
            except:
                continue
        
        return response

    speaker_name = speaker_names[speaker_num]

    for i, log_ in enumerate(dialog):
        if i == 0:
            continue
        speaker_n = int(list(log_.keys())[0])
        message = list(log_.values())[0]
        dialog[i] = {f"{speaker_names[speaker_n]}: {message}"}

    profiles = {
        "Pat": "You are a world-class VC who can evaluate the best startup ideas. If you hear the idea, you should criticize and play devil's advocate.",
        "Lily": "You are a world-class VC who can evaluate the best startup ideas. If you hear the idea, you should encourage and give positive feedback.",
        "Steve": "You are a world-class researcher who can help to build the best startup. If you hear the idea, you research the idea on the internet and give feedback based on concrete facts",
        "Casey": "You are a honest potential customer who can give feedback to the idea. If you hear the idea, you should give honest feedback.",
    }
    
    system_message = f"""
    Here is a world-class startup team.
    They do their very best to generate detailed business plan.
    There are 4 members in the team: 
    {profiles}

    The team is trying to generate detailed business plan from the abstract idea.
    The service name is {ds.log["service_name"]}
    The problem they are trying to solve is {ds.log["problem"]}
    The abstract idea is {ds.log["service_idea"]}

    The team has discussed the following ideas so far:
    {dialog}
    """

    if ds.human_input != "":
        system_message += f"CEO: {ds.human_input}\n"

    user_message = f"This is turn of {speaker_name} to speak. {speaker_name}, please give your feedback to the idea. Keep it short, less than 5 sentences."

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": SystemMessage(content=system_message),
    }

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )

    result = agent({"input": user_message})

    response = {"speaker": speaker_num, 
                "contents": result["output"],
                "is_finished": len(speaker_list) == 0}

    return response

@app.post("/generate_business_plan/")
async def generate_business_plan(bp: BusinessPlanSource):
    system_message = f"""
    You are a world-class startup business developer.
    Based on the discussion, you should generate detailed business plan.
    Here is the discussion so far:
    {bp.log["dialog"]}
    """
    user_message = """
    Please generate detailed business plan based on the discussion. 
    Reply only in json with the following format:

    {
        \"business_plan\": {
            \"business_plan\":  \"name of business plan\",
            \"executive_summary\": \"describe executive summary\",
            \"problem_statement\": \"describe clear problem statement \",
            \"solution\": \" write solution\",
            \"target_market\": \"describe target market\",
            \"revenue_model\": \"describe revenue model\",
            \"GoToMarket_strategy\": \"describe GoToMarket strategy\",
            \"competitive_analysis\": \"describe competitive analysis\",
            \"operaion_plan\": \"describe operation plan\",
            \"conclusion\": \"describe conclusion\",
        },
    }
    
    """
    while True: 
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )

        response = response["choices"][0]["message"]["content"]
        try: 
            business_plan = json.loads(response)
            break
        except: 
            continue
    
    return business_plan

