from functools import partial
from .utils import openai_completion


def data_analyst(
    question: str,
    openai_api_key: str,
    data_fetcher,
) -> dict:
    action_data = {}
    llm = partial(
        openai_completion,
        api_key=openai_api_key,
        temperature=0,
        model="text-davinci-003",
        max_tokens=2_000,
    )
    plan = Plan(question=question, llm=llm)
    for action in plan:
        if action["tool"] == "FooBar DB":
            obs = query_data(
                question=action["input"], data_fetcher=data_fetcher, llm=llm
            )
            action_data["data"] = obs["df"]
            plan.add_information(obs["df_obs"])
        elif action["tool"] == "Plotter":
            input_context = plan.prompt.replace(plan_prompt_intro, "").strip()
            code = plot_data(
                input_context=input_context,
                question=action["input"],
                llm=llm,
            )
            action_data["code"] = code
            plan.add_information(code)
        else:
            pass
    return dict(action=action, action_data=action_data)


class Plan:
    """Class for planning the actions to take to answer a questions"""

    def __init__(self, question: str, llm):
        self.prompt = plan_prompt_intro + plan_prompt_thoughts.format(question=question)
        self.llm = llm

    def next_step(self, stop: str = "\nObservation:") -> str:
        action = self.llm(self.prompt, stop=stop, log_completion="plan")
        self.prompt = self.prompt + action
        return action

    def add_information(self, observation: str) -> None:
        self.prompt = self.prompt + "\nObservation: " + observation + "\nThought:"

    def __iter__(self, max_steps=5) -> str:
        for i in range(max_steps):
            step = self.next_step()
            thought, action, action_input = (
                step.split("\n")[:3] if step else ["", "", ""]
            )
            action_out = dict(
                tool=action.split(":")[-1].strip(),
                input=action_input.split(":")[-1].strip(),
            )
            if "\nFinal Answer:" in step or step == "":
                break
            else:
                yield action_out
        yield action_out


def query_data(question: str, data_fetcher, llm) -> dict:
    prompt = data_prompt.format(tables_info=data_fetcher.tables_info, question=question)
    sql_statement = llm(prompt, stop="\nSQLResult:", log_completion="data")
    df = data_fetcher.exec_sql(sql_statement)
    df_observation = "\n" + df.head().to_markdown()
    return dict(df_obs=df_observation, df=df.to_json())


def plot_data(input_context: str, question: str, llm) -> str:
    prompt = plot_prompt.format(input_summary=input_context, question=question)
    plot_code = llm(prompt=prompt, stop="\nfig.show()", log_completion="plot")
    return plot_code


#############################
###   Prompt templates    ###
#############################
plan_prompt_intro = """Answer the following questions as best you can. You have access to the following tools:

Plotter: useful for when you need to show a graph or plot a figure.
FooBar DB: useful for when you need to answer questions about FooBar or need to get data to show in graph. Input should be in the form of a question containing full context

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Plotter, FooBar DB]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Action: the action to take, should be one of [Text, Plot] 
Final Answer: the final answer to the original input question

Begin!
"""
plan_prompt_thoughts = """Question: {question}
Thought:"""


plot_prompt = """I want you to act as a data scientist and code for me. Given an input question and an input summary please write code for visualizing the data in the dataframe df.

The figure should clearly and effectively communicate the information in the data, and should be visually appealing. Please use Plotly's features such as annotations, color scales, and subplots as appropriate to enhance the figure's readability and impact.

Use the following format:

Question: "Question here"
Code: "Code to run here"

### Input Summary
{input_summary}
###


Question: {question}
Code:"""

data_prompt = """Given an input question, first create a syntactically correct sqlite query to run, then look at the results of the query and return the answer. You can order the results by a relevant column to return the most interesting examples in the database at the top.

Never query for all the columns from a specific table, only ask for the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{tables_info}

Question: {question}
SQLQuery:"""
