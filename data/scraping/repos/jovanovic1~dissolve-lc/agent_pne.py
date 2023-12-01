from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.tools import Tool,StructuredTool, tool
from langchain import LLMMathChain
from WebActionIdentifier import WebActionIdentifier
from ReturnHTMLCode import ReturnHTMLCode
import DissolveAgent

# import ReturnHTMLCode

#dev tools

#Tools
search = SerpAPIWrapper(serpapi_api_key='9792e14cdb37c51c2a8210c8d66eefdb811488b691f3f8b66da3134c515bf04d')
llm = OpenAI(openai_api_key='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',temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
actioner = WebActionIdentifier()
returner = ReturnHTMLCode()
tools = [
    Tool(
        name = "web_action_identifier",
        func=actioner._run,
        description="the input for this tool is the user query and the html code of the website, it returns the action to be taken on the website"
    ),
    Tool(
        name = "fetch_html_code",
        func=returner._run,
        description = "this will help fetch the html_code of the page user is currenly viewing"
    )
]

#model
model = ChatOpenAI(openai_api_key='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',temperature=0)

#planner
planner = load_chat_planner(model)

#executor
executor = load_agent_executor(model, tools, verbose=True)

#agent
agent2 = PlanAndExecute(planner=planner, executor=executor, verbose=True)

base_prompt = """
You are an intelligent site navigator who is helping a user navigate a website. You will be given a query by
the user about what he wants to achieve on the website for example: add a product to his cart, or return their last order, or
filter the products by price. You can also use the fetch_html_code tool to fetch the html code of the page the user is currently viewing.
Your job is to provide the element of the page and the action to be taken on that element to achieve the user's goal. The output can 
be sent by using the web_action_identifier tool.

if the user's job cannot be fullfiled from the current webpage then it could be possible that you have to go through a series of 
webpages to reach the required page where the final element action needs to be taken. Your job is also to navigate the user through the
webpages. 
"""
print(agent2.executor.agent.llm_chain.prompt.template)

# with open("html_code.txt", "w") as f:
#     f.write(str(agent.templ))
# output = agent.run(base_prompt + "User: I want to buy a keyboard compatible with iPad.")
# print(output)
# actioner = WebActionIdentifier()
# output = actioner._run(query="I want keyboard compatible with iPad", html_code=html)
# print(output)