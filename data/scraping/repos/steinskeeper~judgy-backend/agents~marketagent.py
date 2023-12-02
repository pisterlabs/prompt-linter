import asyncio
from fastapi import APIRouter
from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain.llms import VertexAI
from pymongo import MongoClient
from bson.objectid import ObjectId
import vertexai
from vertexai.language_models import TextGenerationModel
search = DuckDuckGoSearchRun()

router = APIRouter()
client = MongoClient("mongodb://localhost:27017/")
db = client["judgy"]


@router.get("/market-agent")
async def marketAgent_endpoint():
    await asyncio.sleep(5)
    return {"message": "Hello from Market Agent"}


async def invoke_market_agent(project_id: str, idea: str):
    vertexai.init(project="lofty-bolt-383703", location="us-central1")
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 1000,
        "top_p": 0.8,
        "top_k": 40
    }
    for x in db.hackathons.find():
        technologies = x["technologies"]
        theme = x["theme"]
        break

    llm = VertexAI()
    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description="useful for when you need to ask with search",
        )
    ]

    marketQuestion = [
        "Who is the target audience of this idea?",
        "What is the potential of this idea?",
        "What is the market size of this idea?",
        "What are the pitfalls of this idea?",
        "Are there any platforms like the idea, that already exist?"]
    agentAnswers = []

    def getAnswer(question):
        self_ask_with_search = initialize_agent(
            tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True
        )
        prompt = """
        You are a market researcher. You have to answer the question about the idea.
        Idea: {idea}
        Question: {question}
        Rules for answering: 
            1. Use statistical data where ever possible.
            2. Remember to answer like a market researcher.
            3. Answer the question as best you can, in a paragraph.
            4. You must answer in one paragraph. Do not use formatting.
            5. Your paragraph must not have more than 70 words.
        """
        prompt = prompt.format(question=question, idea=idea)
        resp = self_ask_with_search.run(prompt)
        agentAnswers.append(resp)
    for i in marketQuestion:
        getAnswer(i)

    final = []
    for i in range(len(marketQuestion)):
        newval = {"question": marketQuestion[i], "answer": agentAnswers[i]}
        final.append(newval)
    print(final)

    model = TextGenerationModel.from_pretrained("text-bison@001")
    theme_prompt = """
    Themes : {theme}
    Idea: {idea}
    Task : To which of the above themes does the idea belong to?
    Rules: 
    1. The theme must be from the above list
    2. Just give the matched theme and nothing more.
    3. If the idea does not match any of the themes, just say "None"
    """
    theme_prompt = theme_prompt.format(theme=theme, idea=idea)
    response = model.predict(
        theme_prompt,
        **parameters
    )
    finalTheme = response.text
    print("project_id", project_id)
    query = {"_id": ObjectId(project_id)}
    newvalues = {"$set": {"marketAgentAnalysis": final, "theme": finalTheme}}
    temp = db.projects.update_one(query, newvalues)
    print("Market Agent : Task Complete")
