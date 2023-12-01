from langchain.prompts import PromptTemplate

SUMMARY_PROMPT_DESCRIPTION_LONG_TMPL = (
    "CONSTRAINTS:\n"
    "You're a programming expert.\n"
    "Your job is to summarize the problem description step by step\n"
    "And to estimate the expected time complexity and space complexity of the solution\n"
    "Below is the problem information for your reference\n"
    "PROBLEM INFORMATION:\n"
    "--------\n"
    "{problem_info}\n"
    "--------\n"
    "I have provide an existing summary up to a certain point\n"
    "EXISTING_SUMMARY:\n"
    "{existing_summary}"
    "--------\n"
    "CONSTRAINTS:\n"
    "provide two texts below, divided into 1000 tokens\n"
    "The first text is provided to help your memory and is partially reflected in the summary\n"
    "The second text is the one you will need to reference in your summary this time\n"
    "PROBLEM_DESCRIPTION:\n"
    "----(first text)----\n"
    "{first_text}\n"
    "----(first text end)----\n"
    "----(second text)----\n"
    "{second_text}\n"
    "----(second text end)----\n"
    "CONSTRAINTS:\n"
    "The problem description summary must include all logical parts of the problem\n"
    "You must synthesize all of the explanations and information provided to arrive at your conclusion\n"
    "Algorithm type should be used as a tool for powerful reasoning\n"
    "Remember to get the best possible time complexity. Avoid inefficient algorithms (if possible)\n"
    "Don't consider Bruteforce(if possible)\n"
    "If the description given is not in English, it must be translated into English\n"
    "Please write all data in sentence form, not list form.\n"
    "You should respond only as described below\n"

    "RESPONSE FORMAT:\n"
    "--------\n"
    "gpt_problem_summary_description: <summarized problem description>,\n"
    "gpt_time_complexity: <best guess of time complexity given information, problem description. only Big O Notation>,\n"
    "gpt_time_complexity_reason: <reason of time complexity>,\n"
    "gpt_space_complexity: <best guess of spatial complexity given information, problem description. only Big O Notation>,\n"
    "gpt_space_complexity_reason: <reason of spatial complexity>,\n"
    "--------\n"
)

SUMMARY_PROMPT_DESCRIPTION_LONG = PromptTemplate(
    input_variables=["problem_info", "existing_summary", "first_text", "second_text"],
    template=SUMMARY_PROMPT_DESCRIPTION_LONG_TMPL,
)