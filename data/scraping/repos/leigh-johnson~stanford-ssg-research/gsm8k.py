from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.prompt_template import BasePromptTemplate

from llm_programs.prompts.base import BasePrompt, PromptTemplateType
from llm_programs.models import InstructModel

ANSWER_TOKEN = "####"  # indictates the final answer in ground truth

DATA = {
    "id": "gsm8k",
    "name": "Middle school arithmetic problems",
    "task_description": "Answer the following middle school math word problem.",
    "task_description_cot": "Answer the following middle school math word problem, which requires multi-step arithmetic reasoning. Let's think step-by-step.",
    "task_description_with_program": "(Grade school math) Solve the following middle-school arithmetic problems, using Python code to solve intermediate arithmetic calculations. Wrap code in ``` for readability. Store your result as a variable named 'ans' and print(ans) as the final step.",
    "examples_with_thoughts": [
        {
            "input": "Mason is cleaning out all the junk in his attic. 20% of the items are useful, 10% are valuable heirlooms, and 70% are junk. If Mason's attic has 8 useful items in it, how many junk items does it have?",
            "thoughts": """If Mason had a total of x items in his attic, 20% of the items are useful, 10% are valuable heirlooms, and 70% are junk.
We need to figure out what x is, given that 20% of x is 8.
We can do this by dividing 8 by 0.2 (20%) to get 40. This means that there are 40 items in total the attic.
Finally, to find the number of junk items, we need to figure out what 70% of 40 is. We can do this by multiplying 40 by 0.7 to get 28. 
This means that there are 28 junk items in the attic.""",
            "answer": "28 junk items",
        },
        {
            "input": "A gecko eats 70 crickets every three days.  The first day she eats 30% of the crickets. The second day she eats 6 less than the first, and the third day she finishes up the remaining crickets.  How many crickets does she eat on the third day?",
            "thoughts": """On the first day, the gecko eats 30% of 70 crickets, which is 21 crickets.
On the second day, she eats 6 less than that, so she eats 15 crickets.
On the third day, she eats the remaining crickets. That will be 70 - 21 - 15, which is 34.""",
            "answer": "34 crickets.",
        },
        {
            "input": "My new house has 12 medium ceiling lights but I have seen small and large ceiling lights in other rooms. The small ones require 1 bulb, the medium ones require 2, and the large ones need 3 bulbs. How many bulbs should I buy if my wife says she saw twice as many large ceiling lights as medium ceiling lights and ten more small lights than medium ones?",
            "thoughts": """First, we need to figure out how many large and small ceiling lights there are.
We know that there are 12 medium ceiling lights, so if there are twice as many large ones, that means there are 24 large ones.
We also know that there are ten more small ones than medium ones, so that means there are 22 small ones.
Now that we know how many of each type there are, we can figure out how many bulbs we need.
Remember, small ones require 1 bulb, medium ones require 2, and large ones require 3.
That means we need 22 bulbs for the small ones, 24 bulbs for the medium ones, and 72 bulbs for the large ones.
All together, we need 118 bulbs.""",
            "answer": "118 bulbs",
        },
        {
            "input": "Tim buys a cabinet for $1200 and gets a 15% discount. How much did he pay?",
            "thoughts": "To calculate the discount Tim gets, we find 15%% of 1200. This is 180. Subtracting the discount amount from 1200 gets us 1020. Thus, Tim paid 1020.",
            "answer": "$1020",
        },
        {
            "input": "Grant scored 10 points higher on his math test than John. John received twice as many points as Hunter who scored a 45 on his math test.  What was Grant's test score?",
            "thoughts": "Hunter scored a 45 on his math test. John received twice as many points as Hunter. Thus John got 90. Grant scored 10 points higher on his math test than John. So Grant got a 100 on the test.",
            "answer": "100 points",
        },
    ],
    "examples_with_tools": [
        {
            "input": "A toy manufacturer receives an order for 400 toys. 5 workers are available to work on the order. 2 of the workers produce 6 toys an hour, and another 2 workers produce 4 toys an hour. They all work on the order during their 10-hour shift, and by the end of their shift the manufacturer still needs another 20 toys to be able to ship the order. How many toys per hour does the fifth worker produce?",
            "actions": """Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
num_toys_ordered = 400
num_workers = 5
toys_produced_per_hour_by_worker1 = 6
toys_produced_per_hour_by_worker2 = 6
toys_produced_per_hour_by_worker3 = 4
toys_produced_per_hour_by_worker4 = 4
toys_produced_per_hour_by_worker5 = Symbol('toys_produced_per_hour_by_worker5', positive=True)
hours_worked = 10
toys_produced = num_toys_ordered-20
toys_produced_by_all_workers = ( toys_produced_per_hour_by_worker1 + toys_produced_per_hour_by_worker2 + toys_produced_per_hour_by_worker3 + toys_produced_per_hour_by_worker4 + toys_produced_per_hour_by_worker5) * hours_worked
solution = solve_it(toys_produced_by_all_workers - toys_produced, toys_produced_per_hour_by_worker5)
ans = solution[toys_produced_per_hour_by_worker5]
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 18
Q3: [add unit] Add the appropriate unit to the final answer.
#3: 18 toys
Q3: [EOQ]""",
            "answer": "18 toys",
        },
        {
            "input": "If two trains depart from a station in opposite directions, and one train is traveling 60 miles an hour while the other is traveling half that distance per hour, how far apart are they from each other after 3 hours?",
            "actions": """Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
speed_of_first_train = 60
speed_of_second_train = 30
distance_apart = speed_of_first_train * 3 + speed_of_second_train * 3
ans = distance_apart
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: 270
Q3: [add unit] Add the appropriate unit to the final answer.
#3: 270 miles
Q4: [EOQ]""",
            "answer": "270 miles",
        },
    ],
}

# TODO
# EXAMPLE_TOOL_PROMPT_TEMPLATE = PromptTemplate(
#     input_variables=["input", "actions", "answer"],
#     template="Input: {input}\n{actions}\nFinal Answer: {answer}",
# )

# FEW_SHOT_TOOL_PROMPT_TEMPLATE = FewShotPromptTemplate(
#     examples=DATA["examples_with_tools"],
#     example_prompt=EXAMPLE_TOOL_PROMPT_TEMPLATE,
#     prefix=DATA["task_description_with_program"],
#     suffix="Question: {input}",
#     input_variables=["input"],
# )

# TODO
# EXAMPLE_COT_PROMPT_TEMPLATE = PromptTemplate(
#     input_variables=["input", "thoughts", "answer", "cot_prompt"],
#     template="Input: {input}\n Answer: Let's think step-by-step.\n{thoughts}\nFinal Answer: {answer}",
# )

# FEW_SHOT_COT_PROMPT_TEMPLATE = FewShotPromptTemplate(
#     examples=DATA["examples_with_thoughts"],
#     example_prompt=EXAMPLE_COT_PROMPT_TEMPLATE,
#     prefix=DATA["task_description_with_thoughts"],
#     suffix="Question: {input}",
#     input_variables=["input"],
# )

# Return a callable here
# If these are instantiated as singletons, they leak stake


DIRECT_PROMPT_TEMPLATE = PromptTemplate(
    partial_variables=dict(),
    input_variables=["question", "task_description"],
    template="""{task_description}
Question: {question}
Answer:
""",
)

EXAMPLE_DIRECT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["input", "answer"],
    template="Input: {input}\nFinal Answer: {answer}",
)

FEW_SHOT_DIRECT_PROMPT_TEMPLATE = FewShotPromptTemplate(
    examples=DATA["examples_with_thoughts"],
    example_prompt=EXAMPLE_DIRECT_PROMPT_TEMPLATE,
    prefix=DATA["task_description_cot"],
    suffix="Question: {input}",
    input_variables=["input"],
)


# PROMPT_SELECTOR = ConditionalPromptSelector(
#     default_prompt=DIRECT_PROMPT_TEMPLATE,
#     conditionals=[(is_zero_shot_direct, DIRECT_PROMPT_TEMPLATE)],
# )


class Gsm8kPrompt(BasePrompt):
    def parse_final_answer(self, text: str) -> str:
        """
        Parse final result line in GSM8k dataset

        Example input:
        Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
        #### 72

        Example output:
        72
        """
        return text.split(ANSWER_TOKEN)[-1].strip()

    def task_description(self) -> str:
        if self.prompt_template_type is PromptTemplateType.DIRECT:
            return DATA["task_description"]
        elif self.prompt_template_type is PromptTemplateType.COT:
            return DATA["task_description_cot"]
        elif self.prompt_template_type is PromptTemplateType.PROGRAM:
            return DATA["task_description_with_program"]

        raise NotImplementedError(
            f"Task description for {self.prompt_template_type} is not yet implemented, please add to prompts/gsm8k.py"
        )

    def few_shot_cot_prompt(self, num_examples: int) -> BasePromptTemplate:
        examples = []
        for i in range(0, num_examples):
            example_q = DATA["examples_with_thoughts"][i]["input"]
            example_t = DATA["examples_with_thoughts"][i]["thoughts"]
            example_a = DATA["examples_with_thoughts"][i]["answer"]
            example = f"""Question: {example_q}
Answer: Let's think step-by-step.
{example_t}
Final Answer: {example_a}
"""
            examples.append(example)
        examples = "\n".join(examples)
        return PromptTemplate(
            validate_template=True,
            partial_variables={"task_description": self.task_description(), "examples": examples},
            input_variables=["question"],
            template="""{task_description}
{examples}
Question: {question}
Answer: Let's think step-by-step.
""",
        )

    def zero_shot_cot_prompt(self) -> BasePromptTemplate:
        return PromptTemplate(
            validate_template=True,
            partial_variables={"task_description": self.task_description()},
            input_variables=["question"],
            template="""{task_description}
Question: {question}
Answer:""",
        )

    def zero_shot_direct_prompt(self) -> BasePromptTemplate:
        return PromptTemplate(
            validate_template=True,
            input_variables=["question"],
            template="""Question: {question}
Answer:""",
        )

    def zero_shot_program_prompt(self, task_description="") -> BasePromptTemplate:
        if self.instruct_model in [InstructModel.CODELLAMA_7B_INSTRUCT_HF, InstructModel.CODELLAMA_7B_PYTHON_HF]:
            # based on: https://huggingface.co/blog/codellama#conversational-instructions
            return PromptTemplate(
                validate_template=True,
                partial_variables={"task_description": self.task_description()},
                input_variables=["question"],
                template="""<s>[INST] <<SYS>>
{task_description}
<</SYS>>
{question}
[/INST]""",
            )
        raise NotImplementedError

    def few_shot_program_prompt(self, num_examples: int, task_description="") -> BasePromptTemplate:
        raise NotImplementedError


PROMPT = Gsm8kPrompt
