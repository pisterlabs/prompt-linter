from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# examples, prompt template, combine

input_variables = ["problem", "solution"]

examples = [
    { "problem": "2 + 2", "solution": "4" },
    { "problem": "5 * 3", "solution": "15" },
]

prompt_template_template = """
Problem: {problem}
Solution: {solution}\n
"""

prompt_template = PromptTemplate(input_variables=input_variables, template=prompt_template_template)

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt_template,
    prefix="""
You are an intelligent math teacher. Give the correct solution to the math problem.
""",
    suffix="Problem: {problem}\nSolution:",
    input_variables=["problem"],
    example_separator="\n\n",
)

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

problems = [
    "6 * 6",
    "2 + 2",
    "5 * 3",
]

for problem in problems:
    print(chain.run(problem=problem))
