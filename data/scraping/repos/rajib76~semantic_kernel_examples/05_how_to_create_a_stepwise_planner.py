# Borrowed from semantic kernel github
import asyncio

from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.planning import StepwisePlanner
from semantic_kernel.planning.stepwise_planner.stepwise_planner_config import (
    StepwisePlannerConfig,
)
import semantic_kernel as sk

kernel = sk.Kernel()
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service("gpt-3.5", OpenAIChatCompletion("gpt-3.5-turbo-16k", api_key, org_id))
class WebSearchEngineSkill:
    """
    A search engine skill.
    """
    from semantic_kernel.orchestration.sk_context import SKContext
    from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter

    def __init__(self, connector) -> None:
        self._connector = connector

    @sk_function(
        description="Performs a web search for a given query", name="searchAsync"
    )
    @sk_function_context_parameter(
        name="query",
        description="The search query",
    )
    async def search_async(self, query: str, context: SKContext) -> str:
        query = query or context.variables.get("query")[1]
        result = await self._connector.search_async(query, num_results=5, offset=0)
        return str(result)


from semantic_kernel.connectors.search_engine import BingConnector

BING_API_KEY = sk.bing_search_settings_from_dot_env()
connector = BingConnector(BING_API_KEY)
kernel.import_skill(WebSearchEngineSkill(connector), skill_name="WebSearch")

from semantic_kernel.core_skills.math_skill import MathSkill
from semantic_kernel.core_skills.time_skill import TimeSkill

kernel.import_skill(TimeSkill(), "time")
kernel.import_skill(MathSkill(), "math")

planner = StepwisePlanner(
    kernel, StepwisePlannerConfig(max_iterations=10, min_iteration_time_ms=1000)
)

ask = """How many total championships combined do the top 5 teams in the NBA have?"""

plan = planner.create_plan(goal=ask)

result = asyncio.run(plan.invoke_async())
print(result)
for index, step in enumerate(plan._steps):
    print("Step:", index)
    print("Description:",step.description)
    print("Function:", step.skill_name + "." + step._function.name)
    if len(step._outputs) > 0:
        print( "  Output:\n", str.replace(result[step._outputs[0]],"\n", "\n  "))