from flask import Flask, render_template, request, Response
from flask_cors import CORS
from flask import jsonify
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import Anthropic
from tempfile import TemporaryDirectory
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import SequentialChain
from agent import plan_execute
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools, initialize_agent,AgentType

working_directory = TemporaryDirectory()

load_dotenv()

app = Flask(__name__)
#CORS(app, resources={r"/*": {"origins": "*"}})

llm = Anthropic(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=0)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/data', methods=['GET'])
def get_data():
    data = {"message": "Hello from Flask!"}
    return jsonify(data), 200


@app.route('/api/plan', methods=['POST'])
def basic_impl():
    """Take the context and starts the sequence chain"""
    data = request.get_json()
    idea = data['idea']
    problemDefinition = data['problemDefinition']
    targetAudience = data['targetAudience']
    constraints = data['constraints']
    solutionOverview = data['solutionOverview']
    metricsAndGoals = data['metricsAndGoals']

    analysis_chain = analysis_bot()
    business_planner_chain = business_planner()
    project_chain = project_planner()
    req_detailer_chain = req_detailer()
    risk_chain = risk_assess()
    overall_chain = SequentialChain(chains=[analysis_chain,req_detailer_chain,business_planner_chain, project_chain,risk_chain],input_variables=["idea","problemDefinition","metricsAndGoals","targetAudience","constraints","solutionOverview"],output_variables=["requirements_USPs", "requirements_details", "project_plan","risk_assessment"],verbose=True)
    answer = overall_chain(
        {
            "idea":idea,
            "problemDefinition":problemDefinition,
            "metricsAndGoals":metricsAndGoals,
            "targetAudience":targetAudience,
            "constraints":constraints,
            "solutionOverview":solutionOverview,
        }
    )

    return jsonify({'answer': answer})


# ############### CHAINS #####################

def analysis_bot():
    """Takes context and generates requirements and USPs"""
    llm = Anthropic(streaming=True,callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),temperature=0.7)
    template = """You are a Software Development analysis bot. Your job is to take an idea, a problem for certain target audience and based on the
    target audience you need to analyse the given information, you should generate functional requirements and non functional requirements.
    You should also identify USPs (unique selling points) this software should have to compete in the market.

    Here is the idea: {idea} and here is the problem: {problemDefinition} for the following target audience: {targetAudience}

    Here are the constraints: {constraints}. If they are not given, assume there are no constraints. 

    So broadly, the solution overview is: {solutionOverview} with the metrics and goals of the project: {metricsAndGoals}

    These are requirements and USPs based on the given data: (give the answer rich text format)
    """
    prompt_template = PromptTemplate(input_variables=["idea","problemDefinition","metricsAndGoals","targetAudience","constraints","solutionOverview"], template=template)
    analysis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="requirements_USPs")
    return analysis_chain


def req_detailer():
    """Take a requirement and makes it more detailed"""
    llm = Anthropic(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=0)
    template = """You are a senior software developer bot who can write amazing code. Now, given the requirements {requirements_USPs}, return the requirements in a json
    requirement: To do something, priority : low or med or high, time_to_complete: in days. Make sure you dont end the answer in the middle of a sentence.
    it is a json object.

    Make sure to keep the format simple to avoid errors and make sure to cover all the requirements
    """
    prompt_template = PromptTemplate(input_variables=["requirements_USPs"], template=template)
    req_prior = LLMChain(llm=llm, prompt=prompt_template, output_key="requirements_details")
    return req_prior

def business_planner():
    """Takes detailed req and generates business plan"""
    llm = Anthropic(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), temperature=0)
    business_bot = """
    You are a really smart business bot well-versed in a various business fields. Given requirement details {requirements_details}
    the idea, you should do the following things:
    Things to do:
    Briefy summarise current market landscape
    Financial projection
    Marketing and Sales Strategy
    Customer Support and Service Strategy

    Use best practices to answer and don't fabricate answers. Use truthful knowledge.

    """
    prompt_template = PromptTemplate(input_variables=["requirements_details"], template=business_bot)
    business_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="business_plan")
    return business_chain


def project_planner():
    """Takes requirements and generates project plan"""
    llm = Anthropic(streaming=True,callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),temperature=0.5)
    template = """You are a software developement planning agent.
    Given the requirements and business plan , develop a proper project plan that identifies, prioritizes, and assigns the tasks and
    resources required to build the project

    Business Plan:
    {business_plan}
    Project Plan in the order of priority:

    """
    prompt_template = PromptTemplate(input_variables=["business_plan"], template=template)
    project_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="project_plan")

    return project_chain


def risk_assess():
    
    template = """You are a software development risk assessment tool that designs the system for the 
    following software idea and requirements along with the following project plan {project_plan}. I want you to write up a risk assessment tool.

    Risks Involved and mitigation:
    """
    prompt_template = PromptTemplate(input_variables=["project_plan"], template=template)
    risk_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="risk_assessment")

    return risk_chain







def add_Item(query: str) -> str:
    import os, json, requests
    url = "https://api.monday.com/v2"

    headers = {
    'Authorization': os.getenv("MONDAY_API_KEY"),
    'Content-Type': 'application/json'
}

    payload = json.dumps({
        "query": f"mutation {{create_item (board_id: 1216344722, group_id: \"topics\", item_name: \"{query}\") {{id}}}}"
    })

    response = requests.request("POST", url, headers=headers, data=payload)
    
    if response.status_code == 200:
        return "Item added successfully"
    else:
        return f"Request failed: {response.text}"


############# AGENTS ###############################

@app.route('/api/agent_monday', methods=['POST'])
def market_analysis_agent_search():
    """ Takes an area to research and does so using internet"""
    data = request.get_json()
    project_plan = data['project_plan']
    requirements = data['requirements_USPs']


    search = SerpAPIWrapper()
    duck_search = DuckDuckGoSearchRun()

    tools = [
        Tool.from_function(
            func=add_Item,
            name="Add Item to Monday.com",
            description="Useful for when you need to add a new item to Monday, input is the name of the new item",
        ),
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,handle_parsing_errors="Check your output and make sure it conforms!")
    market_agent = plan_execute(tools)

    prompt = f"""You are a software developement planning agent.
    Given the requirements, develop a proper project plan by creating tasks. add the tasks to monday.com using the tool provided.
    requirements:
    {requirements}
    Business Plan:
    {project_plan}
    Project Plan in the order of priority:
    """
    

    market_agent.run(prompt)

    return jsonify({'answer': "Project plan created and added to monday.com"})
    

# @app.route('/basic', methods=['POST'])
# def basic_impl():
#         data = request.get_json()
#         context = data['context']
#         prompt = f"""
#         You are an AI smart agent that helps users by generating  
#         Functional Requirements, Non-Functional Requirements. All of the them should be based on the following information 
#         but Don't ask the information that is already given:
#         {context}
#         Make the the answer is the form of JSON object with the following format:
#         'action': 'the action ',
#         'action_input': 'the input of the action',
#         Try to deduce as much as possible from the data but if you think you dont have information about something you cant find using the tools provided, ask for
#         human feedback. Don't ask the information that is already given
#         """

#         agent = plan_execute()

#         answer = agent.run(prompt)

#         return jsonify({'answer': answer})

# @app.route('/requirements', methods=['POST'])
# def fun_non_fun_requirements_agent():
#     data = request.get_json()
#     context = data['context']

#     prompt = f"""
#     I want you to write Functional Requirements and Non functional Requirements for the following application idea:  APP IDEA STARTS HERE 
#     {context} APP IDEA ENDS HERE
#     The requirements you write should be clear, concise, and complete.
#     You should follow industry-standard software engineering practices when writing the requirements.

#     Try to deduce as much as possible from the data 
#     but if you think you dont have information about something you cant find using the tools provided

#     Make sure to keep the format simple to avoid errors
#     """

#     # requirements_agent = plan_execute()
#     requirements_agent = plan_execute()

#     answer = requirements_agent.run(prompt)

#     return jsonify({'answer': answer})


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(port=4000)
