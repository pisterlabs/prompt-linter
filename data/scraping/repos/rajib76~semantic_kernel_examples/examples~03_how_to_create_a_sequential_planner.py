import asyncio

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.core_skills import TextSkill
from semantic_kernel.planning import SequentialPlanner, Plan

with open("./prompts/sk_seq_prompt", "r") as f:
    PROMPT = f.read()

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service("gpt-3.5", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))
skills_directory = "../skills/"
writer_skill = kernel.import_semantic_skill_from_directory(skills_directory, "WriterSkill")
# writer_skill = kernel.import_semantic_skill_from_directory(skills_directory, "WriterSkill")
summarize_skill = kernel.import_semantic_skill_from_directory(skills_directory, "SummarizeSkill")
text_skill = kernel.import_skill(TextSkill(), "TextSkill")
# sk_prompt = """
# {{$input}}
#
# Rewrite the above in the style of Shakespeare.
# """
# shakespeareFunction = kernel.create_semantic_function(sk_prompt, "shakespeare", "ShakespeareSkill",
#                                                       max_tokens=2000, temperature=0.8)
ask = """
Tomorrow is Valentine's day. I need to come up with a few date ideas.
Convert the text to lowercase, summarize the text and then convert to french."""


planner = SequentialPlanner(kernel,prompt=PROMPT)

sequential_plan = asyncio.run(planner.create_plan_async(goal=ask))

# for step in sequential_plan._steps:
#     print(step.description, ":", step._state.__dict__)
#
result = asyncio.run(sequential_plan.invoke_async())
print("final result is ", result)
#
# print(result)