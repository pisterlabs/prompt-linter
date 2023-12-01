from langchain.chat_models import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function

from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.prompts import MessagesPlaceholder
from langchain.schema.agent import AgentFinish
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

from llm_tools.GoogleRoutes import get_routes
from llm_tools.HumetroFare import get_fares
from llm_tools.HumetroWebSearch import get_search
from llm_tools.HumetroSchedule import get_schedule
from llm_tools.StationDistanceTool import get_distance
from llm_tools.HumetroWikiSearch import get_wiki

import langchain
langchain.debug = True

# result = get_wiki.run({"query":"정기승차권은 어떻게 사죠?"})
# print(result)

# result = get_distance.run({"departure":"하단", "destination": "서면"})
# print(result)

# result = get_schedule.run({"station_name":"하단"})
# print(result)

# result = get_routes.run({"departure": "부산역", "destination": "해운대해수욕장"})
# print(result)

# result = get_fares.run({"ages":["어린이", "성인"]})
# print(result)

# result = get_search.run({"query":"문화행사"})
# print(result)


def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "get_wiki": get_wiki,
            "get_distance": get_distance,
            "get_schedule": get_schedule,
            "get_routes": get_routes,
            "get_fares": get_fares,
            "get_search": get_search,
        }
        return tools[result.tool].run(result.tool_input)


def run_agent(user_input):
    intermediate_steps = []
    while True:
        result = chain.invoke({
            "input": user_input,
            "agent_scratchpad": format_to_openai_functions(intermediate_steps)
        })
        if isinstance(result, AgentFinish):
            return result
        tool = {
            "get_wiki": get_wiki,
            "get_distance": get_distance,
            "get_schedule": get_schedule,
            "get_routes": get_routes,
            "get_fares": get_fares,
            "get_search": get_search,
        }[result.tool]
        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))


tools = [get_routes, get_fares, get_search,
         get_schedule, get_distance, get_wiki]
functions = [format_tool_to_openai_function(t) for t in tools]
tools_string = "\n".join([t.name + " : " + t.description for t in tools])
model = ChatOpenAI(model='gpt-3.5-turbo-16k', temperature=0,
                   streaming=True, verbose=True).bind(functions=functions)

system_prompt = f"""너는 부산교통공사 1호선 하단역에서 근무하고 있는 인공지능 역무 보조 어시스턴트야. \
넌 '설동헌 대리(seoldonghun@humetro.busan.kr)'가 개발했어.\
    아래에 고객의 최대한 친절하게 대답해야해. \
    절대로 임의로 대답을 생성하지 말고 주어진 도구를 사용해서 대답해야 해.\
    주어진 도구는 다음과 같아 {tools_string}
    만약 적절한 대답을 찾을 수 없으면. "죄송합니다. 해당 질문에 대한 답변을 찾지 못했습니다." 라고 응답해.\
    """

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


chain = prompt | model | OpenAIFunctionsAgentOutputParser()
agent_chain = RunnablePassthrough.assign(
    agent_scratchpad=lambda x: format_to_openai_functions(
        x["intermediate_steps"])
) | chain

agent_executor_lcel = AgentExecutor(
    agent=agent_chain, tools=tools, verbose=True)

agent_executor_cb = AgentExecutor.from_agent_and_tools(
    agent=agent_chain,
    tools=tools,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)


# from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
# below agent implementation is from langchain/agents/openai_functions_agent/
function_agent = initialize_agent(tools,
                                  ChatOpenAI(model='gpt-3.5-turbo-16k',
                                             temperature=0, streaming=True),
                                  agent=AgentType.OPENAI_FUNCTIONS,
                                  verbose=True,
                                  )

# Override the default prompt for the agent
function_agent.agent.prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="output_language"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
pass

multi_function_agent = initialize_agent(tools,
                                        ChatOpenAI(
                                            model='gpt-3.5-turbo-16k', temperature=0, streaming=True),
                                        agent=AgentType.OPENAI_MULTI_FUNCTIONS,
                                        verbose=True,
                                        )
multi_function_agent.agent.prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

if __name__ == "__main__":
    open_ai_agent.run("Where is the nearest subway station?")
    for q in questions:
        result = agent_executor.invoke({"input": q})
        print(q, " -> ", result['output'])
