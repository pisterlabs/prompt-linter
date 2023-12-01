from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

from langchain.callbacks import get_openai_callback
from langchain.agents import Tool, initialize_agent, AgentType, load_tools
from langchain.utilities import PythonREPL, WikipediaAPIWrapper, GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
import sys
import asyncio
load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

python_repl = PythonREPL()
wolfram = WolframAlphaAPIWrapper()
wikipedia = WikipediaAPIWrapper()
search = GoogleSearchAPIWrapper(k=5)

tools = [
    Tool(
        name="wolfram-alpha",
        func=wolfram.run,
        description="Hasznos amikor tudományos, matematikai, vagy fizikai kérdésekre kell választ kapnod. A bemenet egy keresési kifejezés.",
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Hasznos amikor történelmi, ismeretterjesztő vagy kultúrális kérdésekre kell választ kapnod. A bemenet egy keresési kifejezés."
    ),
    Tool(
        name="google-search",
        func=search.run,
        description="Amikor nincs más válasz, akkor itt biztosan találsz. A bemenet egy keresési kifejezés."
    )
]

power_bor = initialize_agent(llm=llm, tools=tools, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)

def bor_power_mode(prompt):
    return power_bor.run(prompt)


if __name__ == "__main__":
    with get_openai_callback() as callback:
        print(sys.argv[1])
        asyncio.run(bor_power_mode(sys.argv[1]))