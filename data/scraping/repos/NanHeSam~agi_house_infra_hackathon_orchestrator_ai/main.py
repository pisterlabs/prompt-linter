from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.agent_model import FindAgentRequest, FindAgentResponse, Agent, RegisterAgentResponse, FindAgentSingleResponse
from app.agent.langchain_agent import LangChainAgent
from app.agent.llama_index_agent import LlamaIndexAgent
from app.database.pinecone import PineconeClient
from app.database import dynamodb
from app.output_parser.find_agent_result_parser import FindAgentResultParser


app = FastAPI()
llm_agent = LangChainAgent()
pinecone = PineconeClient()
llama_index_agent = LlamaIndexAgent()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health-check")
async def health_check():
    return {"status": "The application is running"}

@app.post("/v1/find-agents", response_model=FindAgentResponse)
async def find_agents(request: FindAgentRequest):
    # look up all the agents
    agents = dynamodb.get_all_agents()
    agent_lookup = {agent['agent_name']: agent for agent in agents}
    if not agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    agent_list = [Agent(**agent) for agent in agents]
    # pass all agent to llm with template
    output_str = llm_agent.query_result(user_query=request.instruction, agents=agent_list)
    print(f'output string: {output_str}')
    # parse llm output
    agents_and_goals = FindAgentResultParser.parse(output_str)
    response = FindAgentResponse()
    # put parsed llm output into format for response
    for agent_and_goal in agents_and_goals:
        agent_name = agent_and_goal.agent_name
        goal = agent_and_goal.goal
        if agent_name in agent_lookup:
            response.agents.append(FindAgentSingleResponse(agent=agent_lookup[agent_name], goal=goal))
        else:
            raise ValueError("agent %s not in the agent list", agent_and_goal.agent_name)   
    
    return response


@app.post("/v2/find-agents", response_model=FindAgentResponse)
async def find_agents(request: FindAgentRequest):
    # ask llm to breakdown tasks
    llm_response = llm_agent.query_simple_tasks_breakdown(instruction=request.instruction, context="")
    # parse llm output into task list
    tasks: list[str] = FindAgentResultParser.parse_to_tasks(llm_response)
    # for each task: 
    response = FindAgentResponse()
    for task in tasks:
        #   send to pinecone for a match
        agent: Agent = pinecone.query_pinecone(task)
        agent = Agent(**agent[0]['metadata'])
        #   pass the match to llm and ask how to achieve it
        why_use_agent = llm_agent.query_find_reason(tool_description=agent.description, task_description=task)
        # synthesize result into response format
        response.agents.append(FindAgentSingleResponse(agent=agent, goal=why_use_agent))
    return response

@app.post("/v3/find-agents", response_model=FindAgentResponse)
async def find_agents(request: FindAgentRequest):
    # ask llm to breakdown tasks
    llm_response = llm_agent.query_simple_tasks_breakdown(instruction=request.instruction, context=request.context)
    # parse llm output into task list
    tasks: list[str] = FindAgentResultParser.parse_to_tasks(llm_response)
    # for each task:
    response = FindAgentResponse()
    for task in tasks:
        #   send to pinecone for a match
        single_task_agent = llama_index_agent.query(task)
        response.agents.append(single_task_agent)
    return response

@app.post("/v1/register-agent", response_model=RegisterAgentResponse)
async def register_agent(agent: Agent):
    message = dynamodb.upsert_agent(
        agent_name=agent.agent_name,
        agent_url=agent.agent_url,
        endpoint=agent.endpoint,
        description=agent.description,
        metadata=agent.metadata
    )

    return RegisterAgentResponse(message=message)
