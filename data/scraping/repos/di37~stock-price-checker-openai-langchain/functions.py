import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

from utils import *
from openai_functions.helper import *
from openai_functions.models import *

tools = [StockPriceTool(),StockPercentageChangeTool(), StockGetBestPerformingTool()]
# functions = [format_tool_to_openai_function(t) for t in tools]
model = ChatOpenAI(model=MODEL)
open_ai_agent = initialize_agent(
    tools, model, agent=AgentType.OPENAI_FUNCTIONS, verbose=True
)
