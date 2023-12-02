import asyncio

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.planning import BasicPlanner

from skills.identify_prime_skill import IdentifyPrime

# Creating the kernel
kernel = sk.Kernel()
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_chat_service("gpt-3.5", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))

#skills_directory = "./skills"
prime_identifier_skill = kernel.import_skill(IdentifyPrime(),"IdentifyPrime")

goal = "check if {number} is prime"

PROMPT = """
You are a planner for the Semantic Kernel.
Your job is to create a properly formatted JSON plan step by step, to satisfy the goal given.
Create a list of subtasks based off the [GOAL] provided.
Each subtask must be from within the [AVAILABLE FUNCTIONS] list. Do not use any functions that are not in the list.
Base your decisions on which functions to use from the description and the name of the function.
Sometimes, a function may take arguments. Provide them if necessary.
The plan should be as short as possible.
For example:

[AVAILABLE FUNCTIONS]
IdentifyPrime.identify_prime_number
description: Identifies if a number is prime
args:
- input: the number to validate if it is a prime number

[GOAL]
"check if 17 is a prime number"
[OUTPUT]
    {
        "input": 17,
        "subtasks": [
            {"function": "IdentifyPrime.identify_prime_number","args": {"number": 17}}
        ]
    }

[AVAILABLE FUNCTIONS]
{{$available_functions}}

[GOAL]
{{$goal}}

[OUTPUT]
"""

planner = BasicPlanner()

while True:
    number = input("Enter the number which you want to check if it is a prime number:\n\n")
    basic_plan = asyncio.run(planner.create_plan_async(goal=goal.format(number=number), kernel=kernel,prompt=PROMPT))
    print("generated plan ",basic_plan.generated_plan)
    # print("generated prompt ", basic_plan.prompt)
    # print("generated goal " ,basic_plan.goal)
    # print("generated str " ,basic_plan.__str__) # THERE IS A DEFECT, RAISED IT TO SEMANTIC KERNEL

    results = asyncio.run(planner.execute_plan_async(basic_plan, kernel))
    print(results)

