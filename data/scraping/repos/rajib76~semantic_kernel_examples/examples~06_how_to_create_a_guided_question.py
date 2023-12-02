import asyncio
import json
import re

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.planning import SequentialPlanner

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service("gpt-4", OpenAIChatCompletion("gpt-4", api_key, org_id))
skills_directory = "../skills/"
advisor_skill = kernel.import_semantic_skill_from_directory(skills_directory, "AdvisorSkill")

planner = SequentialPlanner(kernel)

ask = "What investments are best for me for retirement planning?"
sequential_plan = asyncio.run(planner.create_plan_async(goal=ask))

# for step in sequential_plan._steps:
#     print(step.description, ":", step._state.__dict__)

result = asyncio.run(sequential_plan.invoke_async())
match = re.search(r"```(json)?(.*)```", str(result), re.DOTALL)
json_str = match.group(2)
json_str = json_str.strip()
parsed = json.loads(json_str)

print(parsed)