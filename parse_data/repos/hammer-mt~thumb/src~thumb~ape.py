from .llm import format_chat_prompt


def build_candidate_prompt(task_description, test_cases, criteria=None):

    if len(criteria) == 0:
        ## https://github.com/langchain-ai/langchain/blob/a830b809f39f05026ad32ca1c33c39da7b2bd160/libs/langchain/langchain/evaluation/criteria/eval_chain.py#L45
        criteria.append("Is the submission helpful, insightful, and appropriate?")

    test_cases_partial = ""
    criteria_partial = ""
    unique_keys = set()

    if len(test_cases) > 0:
        test_cases_partial += "Here are some test case scenarios and their example outputs:\n"
        
        for idx, test_case in enumerate(test_cases):
            reference = test_case["__ref__"]
            del test_case["__ref__"]

            case_str = ""
            for key, value in test_case.items():
                case_str += f"""- {key}: "{value}"\n"""
                unique_keys.add(key)
                
            test_cases_partial += (f"""## Test case {idx+1}\nInput variables:\n{case_str.strip()}\nExpected output:\n{reference}\n\n""")
        test_cases_partial = f"\n\n{test_cases_partial.strip()}"
    
    criteria_partial += "The prompt will be deemed successful if it generates responses that meet the following criteria:\n"

    for criterion in criteria:
        criteria_partial += f"- {criterion}\n"

    if len(unique_keys) > 0:
        for key in unique_keys:
            criteria_partial += f"- {{{{ {key} }}}} is included in the prompt\n"

    criteria_partial = f"\n\n{criteria_partial.strip()}"

    candidate_prompt = f"""Here is the task for which we need to build a prompt template:\n{task_description}{test_cases_partial}{criteria_partial}"""
    
    return candidate_prompt


def build_ape_prompt(candidate_prompt):
    system_prompt = """You're a world-leading expert in AI prompt engineering.
Respond with your optimized prompt, and nothing else. Be creative.
NEVER CHEAT BY INCLUDING SPECIFICS ABOUT THE TEST CASES IN YOUR PROMPT. 
ANY PROMPTS WITH THOSE SPECIFIC EXAMPLES WILL BE DISQUALIFIED.
IF YOU USE EXAMPLES, ALWAYS USE ONES THAT ARE VERY DIFFERENT FROM THE TEST CASES."""
    formatted_candidate_prompt = format_chat_prompt([system_prompt, candidate_prompt])
    return formatted_candidate_prompt