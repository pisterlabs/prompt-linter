import datetime
import json
import os
import asyncio
import traceback
from typing import Iterator

from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage, FunctionMessage

from langchain.chains import create_citation_fuzzy_match_chain

from strategy_ai.ai_core import openai_chat
from strategy_ai.ai_core.data_sets.vector_store import FAISSVectorStore

from strategy_ai.tasks.task_models import TaskData, TaskState, TaskTypeEnum

load_dotenv(verbose=True)

# all asserts should be impossible if the code has no bugs
# functions preceded with a _ are not meant to be called outside of this file


def task_save(task: TaskData, directory: str) -> None:
    """This function will save the results of the task to the given directory.

    Inside the given directory, the a subdirectory will be created and will contain:
    - the task's detailed results (pickle file ".pkl")
    - the task's run history (csv file ".csv")
    - the readable copy of the results (markdown text file ".md").

    Args:
        task: A task.TaskData instance to be saved.

        directory: A string representing the directory to save the results to.

    Returns:
        None

    Raises:
        Exception: If the task is not finished.
    """
    if task.state != TaskState.FINISHED:
        raise Exception(
            "Cannot save the results since the task is not finished")

    new_directory = os.path.join(
        directory, f"{task.task_type.value.name}-{task.id}")
    os.mkdir(new_directory)

    with open(file=os.path.join(new_directory, "runHistory.csv"), mode="w", newline="\n") as f:
        for time, entry in task.run_history:
            print(f'{time},"{entry}"', file=f)

    with open(file=os.path.join(new_directory, "readableResults.md"), mode="w", newline="\n") as f:
        print(task.results_text, file=f)

    with open(file=os.path.join(new_directory, "allTaskData.json"), mode="wb") as f:
        f.write(bytes(task.json(indent=4), encoding="ascii"))


def _task_init_surfacing(task: TaskData, vector_store: FAISSVectorStore, llm=ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)) -> None:
    """This function is only called within task_init() and is used to initialize the tasks where `task.task_type` == `TaskTypeEnum.SURFACING`."""
    objectiveCategories = {
        "Financial": [
            "Increasing Revenue",
            "Cost Reduction",
            "Asset Optimization",
        ],
        "Customer": [
            "Improving the Brand",
            "Customer Service",
            "Product Service/Functionality",
        ],
        "Internal": [
            "Operational Excellence",
            "Product Innovation",
            "Regulatory Compliance",
            "Customer Intimacy",
        ],
        "Enabler": [
            "Strategic Assets",
            "Building a Climate for Action",
            "Attracting, Retaining, and Developing Talent",
        ]
    }
    system_message_prompt_template = """Use the following pieces of context to answer the user's question.
Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.

----------------
{context}
"""
    list_objectives_prompt_template = """Provide a list of this company's objectives addressing {topic}, in an organized format.
If there is a function provided, use the function.
Lastly, aim to identify 2-3 objetives. If you cannot find objectives on the topic of {topic}, list some relevant suggestions based on the context.
"""
    objectives_format = {
        "name": "output_formatted_objectives",
        "description": "given a list of objectives for a business in the format of [Verb] [Outcome] [Definition] [Source], print it to the screen.",
        "parameters": {
            "type": "object",
            "properties": {
                "objectives_list": {
                    "type": "array",
                    "minItems": 1,
                    "description": "a list of all the formatted objectives, including the source file for each objective or a direct quote",
                    "items": {
                        "title": "Objective",
                        "type": "object",
                        "properties": {
                            "Verb": {
                                "type": "string",
                                "decription": "A single word describing an action. A verb in present tense, not present continous tense, just present tense",
                            },
                            "Outcome": {
                                "type": "string",
                                "decription": "One or 2 words which are the outcome of this objective",
                            },
                            "Definition": {
                                "type": "string",
                                "description": "Definition should anser these two questions in two sentences: What is the objective working to achieve? Why is the objective important to the business?",
                            },
                            "Source": {
                                "type": "string",
                                "description": "the complete file name where this objective was found, or directly quoted text",
                            },
                        },
                    },
                },
            },
        },
        "required": ["objectives_list"],
    }

    def topic_messages(topic: str) -> list[dict | list]:
        sys_msg = SystemMessage(content=system_message_prompt_template.format(
            context=vector_store.formatted_context(topic)))
        hmn_msg = HumanMessage(
            content=list_objectives_prompt_template.format(topic=topic))

        return [
            {
                "type": openai_chat.MessageRole.SYSTEM,
                "title": "Context",
                "body": sys_msg.content,
            },
            {
                "type": openai_chat.MessageRole.HUMAN,
                "title": "Human Prompt",
                "body": hmn_msg.content,
            },
            [
                {
                    "type": openai_chat.MessageRole.ASSISTANT,
                    "title": "Text Response",
                    "body": None,
                    "task": None,
                    "coro": llm.apredict_messages(messages=[sys_msg, hmn_msg]),
                },
                {
                    "type": openai_chat.MessageRole.ASSISTANT,
                    "title": "Formatted Response",
                    "body": None,
                    "task": None,
                    "coro": llm.apredict_messages(
                        messages=[sys_msg, hmn_msg],
                        functions=[objectives_format],
                        function_call={"name": "output_formatted_objectives"}),
                },
            ],
        ]

    def category_topics(topics: list[str]) -> dict[str, dict]:
        return {
            topic: {
                "title": f"topic: {topic}",
                "body": topic_messages(topic),
            }
            for topic in topics
        }

    task.detailed_results = {
        "title": "Company Objectives",
        "body": {
            category: {
                "title": f"Category: {category}",
                "body": category_topics(topics),
            }
            for category, topics in objectiveCategories.items()
        }
    }


def _task_init_assessment(task: TaskData, vector_store: FAISSVectorStore, llm=ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)) -> None:
    """This function is only called within task_init() and is used to initialize the tasks where task.task_type == TaskTypeEnum.ASSESSMENT."""
    # For now, the goals are just hard coded. Normally the goals would start as an empty list and the llm and vector store would be used to generate the goals.
    goals = [
        "increase sales revenue by 20% compared to last year (from 10M to 12M)"]
    # these are the prompts that could be used to generate the actions that would help achieve a goal
    business_expert_system_message_template = "You are a business expert and you are helping a company achieve the following goal: {goal}"
    list_actions_prompt_template = "List actions that could be taken to achieve the following goal: {goal}"
    use_formatting_function_prompt = "TIP: Use the {function_name} function to format your response to the user."
    formatted_actions_list = {
        "name": "formatted_actions_list",
        "description": "Use this function to output the formatted list of actions to the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions_list": {
                    "title": "Actions List",
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "required": ["actions_list"],
    }
    providing_context_system_message_template = """Use the following pieces of context to answer the user's question.
Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.

----------------
{context}
"""
    look_into_action_prompt_template = """List all the things that the company is doing or has planned to do to carry out the following action: {action}.
Beside each point answer the following questions:
- will this make a difference (refer to what the company has done in previous years and what their competitors are doing)?
- what additional resources does the company require to carry this out?
- what regulations need to be followed?
- what similar things have competing businesses done?
"""

    # summarize_actions_prompt_template = """Based the actions listed below, provide a short summary indicating whether or not it is feasible for "{goal}" to be achieved. \nActions and pertinent info:\n{actions_and_info}"""
    action_info_with_citations = create_citation_fuzzy_match_chain(llm)

    def action_info(action: str) -> list[dict | list]:
        human_message = HumanMessage(
            content=look_into_action_prompt_template.format(action=action))
        context = providing_context_system_message_template.format(
            context=vector_store.formatted_context(human_message.content))
        # system_message = SystemMessage(content=providing_context_system_message_template.format(
        #     context=vector_store.formatted_context(human_message.content)))
        return [
            {
                "type": openai_chat.MessageRole.SYSTEM,
                "title": "Context",
                "body": context,
            },
            {
                "type": openai_chat.MessageRole.HUMAN,
                "title": "Human Message",
                "body": human_message.content,
            },
            [
                {
                    "type": openai_chat.MessageRole.ASSISTANT,
                    "title": "Text Response",
                    "body": None,
                    "task": None,
                    "coro": action_info_with_citations.arun(question=human_message.content, context=context),
                },
                {
                    "type": openai_chat.MessageRole.ASSISTANT,
                    "title": "Formatted Response",
                    "body": None,
                    "task": None,
                    "coro": None,
                },
            ]
        ]

    def goal_summary(actions: list[str]) -> str:
        return {
            action: {
                "title": f"Action: {action}",
                "body": action_info(action),
            } for action in actions
        }

    async def get_actions(goal: str) -> list[str]:
        """As of now, this assumes that the goal is always: increase sales revenue by X% compared to last year.
        """
        await asyncio.sleep(0.1)
        list_of_actions = [
            "increase headcount",
            "increase price",
            "enter new markets",
            "introduce new products",
            "combination",
            "increase customer retention / decrease churn",
        ]
        # list_of_actions = await llm.apredict_messages([
        #     SystemMessage(
        #         content=business_expert_system_message_template.format(goal=goal)),
        #     HumanMessage(
        #         content=list_actions_prompt_template.format(goal=goal)),
        #     SystemMessage(use_formatting_function_prompt.format(
        #         function_name=formatted_actions_list["name"])),
        # ], functions=[formatted_actions_list], function_call="auto").additional_kwargs.get(
        #     "function_call").get("arguments").get("actions_list")
        return list_of_actions

    task.detailed_results = {
        "title": "Feasibility Assessment on the company's goals",
        "body": {
            goal: {
                "title": f"Goal Assessment for: {goal}",
                "body": goal_summary,
                "task": None,
                "coro": get_actions(goal),
            } for goal in goals
        },
        "task": None,
        "coro": None,
    }


def task_init(task: TaskData, vector_store: FAISSVectorStore, llm: ChatOpenAI) -> None:
    """This function will initialize the task.

    It will call the task specific initialization function depeding on `task.task_type`. 

    The task specific initalizer must set the task state to ready.

    Args:
        `task`: A task.TaskData instance to be initialized. **Required for all tasks.**

        `vector_store`: A vector_store.FAISSVectorStore instance to be used for the task. Required for all tasks.

        `llm`: A llm instance to be used for the task. This should be ChatOpenAI(model="gpt-3.5-turbo-0613") or some model that supports function calling. Required for all tasks.

    Returns:
        None

    Raises:
        Exception: If the task is not in the preparing state before calling this function.
        Exception: If the task state was changed while running the task specific initalizer.
    """
    if task.state != TaskState.PREPARING:
        raise Exception(
            f"Cannot initialize task {task.task_type.value.name}, uuid: {task.id}, state: {task.state}. It needs to be in the preparing state.")

    match task.task_type:
        case TaskTypeEnum.SURFACING:
            _task_init_surfacing(task, vector_store, llm)
        case TaskTypeEnum.ASSESSMENT:
            _task_init_assessment(task, vector_store, llm)
        case _:
            raise Exception(f"invalid task type: {task.task_type}")

    # ensure that the task state was not changed in the task specific initializer
    assert task.state == TaskState.PREPARING

    task.state = TaskState.READY


def _task_generate_results_surfacing(task: TaskData, api_call_timeout: int) -> Iterator[dict]:
    """ Generate the results for the task.
    """
    prefix = "## "
    yield {"type": "message", "body": f"Running task {task.task_type.name}, uuid: {task.id}."}
    yield {"type": "results_text", "body": prefix + task.detailed_results["title"]}

    # creating all the async tasks
    async_event_loop = asyncio.new_event_loop()
    for categoryInfo in task.detailed_results["body"].values():
        for topicInfo in categoryInfo["body"].values():
            # iterate over list of messages for the different responses from the AI
            for message in topicInfo["body"][2]:
                message["task"] = async_event_loop.create_task(
                    message["coro"])

    # generating the results
    for category, categoryInfo in task.detailed_results["body"].items():
        prefix = "### "
        yield {"type": "progress_info", "body": f"{category} objectives complete:"}
        yield {"type": "results_text", "body": prefix + categoryInfo["title"]}
        for topic, topicInfo in categoryInfo["body"].items():
            prefix = "#### "
            yield {"type": "results_text", "body": prefix + topicInfo["title"]}
            prefix = "##### "
            # iterate over the messages for the different responses from the AI
            for aiMessage in topicInfo["body"][2]:
                yield {"type": "results_text", "body": prefix + aiMessage["title"]}
                # wait until the coroutine has completed
                try:
                    aiMessage["body"] = async_event_loop.run_until_complete(
                        asyncio.wait_for(aiMessage["task"], timeout=api_call_timeout))
                except asyncio.TimeoutError:
                    aiMessage["task"].cancel()
                    aiMessage["title"] = "Timeout Error"
                    aiMessage["body"] = "API call timed out."
                    print(traceback.print_exc())
                    print("^^^^^^^^ TIMEOUT ERROR ^^^^^^^^")
                # remove the coroutine and task from the message so that it can be serialized
                del aiMessage["task"], aiMessage["coro"]

                if aiMessage["title"] == "Formatted Response" and isinstance(aiMessage["body"], AIMessage):
                    # try to parse the json, if it fails, try to correct the error and parse again
                    try:
                        json_str = aiMessage["body"].additional_kwargs.get("function_call", {}).get(
                            "arguments")
                        aiMessage["body"] = json.loads(
                            json_str)["objectives_list"]
                    except json.decoder.JSONDecodeError as e:
                        # try to correct the error by replacing all \ with \\, effectively escaping the invalid escape characters
                        if e.msg.startswith("Invalid \\escape"):
                            aiMessage["body"] = json.loads(
                                json_str.replace("\\", "\\\\"))["objectives_list"]
                            print(traceback.print_exc())
                            print("^^^^^^^^ CORRECTED ERROR ^^^^^^^^")
                            print("while trying to parse:\n", json_str)
                        else:
                            raise e
                    yield {"type": "results_text", "body": f"```json\n{json.dumps(aiMessage['body'], indent=4)}\n```"}
                elif aiMessage["title"] == "Text Response":
                    aiMessage["body"] = aiMessage["body"].content
                    yield {"type": "results_text", "body": f"```text\n{aiMessage['body']}\n```"}
                elif aiMessage["title"] == "Timeout Error":
                    yield {"type": "results_text", "body": f"```text\n{aiMessage['body']}\n```"}
                else:
                    assert False, f"invalid aiMessage title: {aiMessage['title']}"

            yield {"type": "progress_info", "body": f"- {topic}"}


def _task_generate_results_assessment(task: TaskData, api_call_timeout: int) -> Iterator[dict]:
    """This function is used to generate the results of the task."""
    prefix = "## "
    yield {"type": "message", "body": f"Running task {task.task_type.name}, uuid: {task.id}."}
    yield {"type": "results_text", "body": prefix + task.detailed_results["title"]}

    # creating all the async tasks
    async_event_loop = asyncio.new_event_loop()
    for goal, goal_dict in task.detailed_results["body"].items():
        # this callback gets the actions about the goal and adds the llm calls to get more detail on the actions to the even loop.
        async def goal_subtasks():
            # before this line goal_dict["body"] is the function goal_summary
            # calling it creates the summary which is a dict of action-info pairs
            goal_dict["body"] = goal_dict["body"](await goal_dict["coro"])
            for action_dict in goal_dict["body"].values():
                # iterate over list of messages for the responses from the AI
                for message in action_dict["body"][2]:
                    if message["coro"] is not None:
                        message["task"] = async_event_loop.create_task(
                            message["coro"])

        goal_dict["task"] = async_event_loop.create_task(
            goal_subtasks())

    for goal, goal_dict in task.detailed_results["body"].items():
        prefix = "### "
        yield {"type": "progress_info", "body": f"{goal}; actions assessed:"}
        yield {"type": "results_text", "body": prefix + goal_dict["title"]}
        # wait for the goal actions to be created
        async_event_loop.run_until_complete(
            asyncio.wait_for(goal_dict["task"], api_call_timeout))
        # to ensure the results are pickleable, remove coroutines and tasks
        del goal_dict["task"], goal_dict["coro"]
        for action, action_dict in goal_dict["body"].items():
            prefix = "#### "
            yield {"type": "results_text", "body": prefix + action_dict["title"]}
            # wait for the action to be assessed
            action_dict["body"][2][0]["body"] = async_event_loop.run_until_complete(
                asyncio.wait_for(action_dict["body"][2][0]["task"], api_call_timeout)).json(indent=4)
            # to ensure the results are pickleable
            del action_dict["body"][2][0]["task"], action_dict["body"][2][0]["coro"]
            yield {"type": "results_text", "body": f'```json\n{action_dict["body"][2][0]["body"]}\n```'}
            yield {"type": "progress_info", "body": f"- {action}"}


def task_generate_results(task: TaskData, api_call_timeout: int = 30) -> Iterator[dict]:
    """This function will return a generator that will yield the dictionaries that can be sent to the frontend.

    Tasks should only be run once, and this function should only be called once per task.

    Args:
        `task`: a TaskData instance to be referenced. This is the task that will be run.

    Returns:
        A generator that will yield the dictionaries that can be sent to the frontend.

    Raises:
        AssertionError: If the task is not in the ready state when this function is called.
        Exception: If the task was set to Finished when the task specific results were being generated.
    """
    assert task.state == TaskState.READY, f"Task is not ready to run. Current state: {task.state}"

    task.message = f"Starting task {task.task_type.value.name}, uuid: {task.id}."
    task.state = TaskState.RUNNING
    task.date_start = datetime.datetime.now()
    yield {"type": "message", "body": task.message}

    # this is where the detailed_results are generated
    # perhaps some messages and progress_info results are generated here also
    # you must ensure that after these results are generated, that the task.detailed_results are pickleable (i.e. no lambda functions)
    match task.task_type:
        case TaskTypeEnum.SURFACING:
            yield from _task_generate_results_surfacing(task, api_call_timeout)
        case TaskTypeEnum.ASSESSMENT:
            yield from _task_generate_results_assessment(task, api_call_timeout)
        case _:
            assert False, f"Task type {task.task_type.value.name} not implemented."

    assert task.state == TaskState.RUNNING, f"Task {task.task_type.value.name} state should not have changed while running, uuid: {task.id}, state: {task.state}."

    yield {"type": "message", "body": f"Finished task {task.task_type.value.name}, uuid: {task.id}. It took {task.date_recent - task.date_start}."}
    task.state = TaskState.FINISHED


def task_generate_results_with_processing(task: TaskData, api_call_timeout: int = 30, save_directory: str | None = None) -> Iterator[dict]:
    """This function will update the task as the results are generated and yielded.

    Args:
        `task`: TaskData instance to be referenced. This is the task that is being run.

        `save_directory`: A string representing the directory to save the results to. If None, the results will not be saved.

    Returns:
        A generator of dicts which are all the results of the task.

        each dict is of the format: {"type": str, "body": str} 
        where type is one of: "results_text", "progress_info", "message"

    Raises:
        Exception: if one of the results generated has the wrong result type.
    """
    for result in task_generate_results(task, api_call_timeout):
        match result["type"]:
            case "results_text":
                task.results_text += result["body"] + "\n"
            case "progress_info":
                task.progress_info += result["body"] + "\n"
            case "message":
                task.message = result["body"]
            case _:
                assert False, f"Unknown result type: {result['type']}."
        task.date_recent = datetime.datetime.now()
        task.run_history.append((task.date_recent, result))
        yield result

    if save_directory is not None:
        yield {"type": "message", "body": f"Saving task {task.task_type.name}, uuid: {task.id}."}
        task_save(task, save_directory)
        yield {"type": "message", "body": f"Saved task {task.task_type.name}, uuid: {task.id}."}


def dict_iter_ndjson_bytes(dict_iter: Iterator[dict]) -> Iterator[bytes]:
    """This function will take an iterator over dictionaries, convert each dictionary to a json string, add a newline, and convert to bytes.
    """
    yield from (bytes(json.dumps(d) + "\n", encoding="ascii") for d in dict_iter)


"""
# Stuff about task 2

I have not thought through how we would do this activity, so this gives you some space to have some fun!
The simple example to work through is that which I talked about on the call today where we want a 20% growth in sales.
Pick a department or function. Examples would be the Sales Department, Customer Service function, or Operations. We need to explore what the increase would be over the previous year for that area, explore possible Objectives, and Key Results then look at the gap in the Key Results from last year's performance and develop recommendations to close the gap.
A slightly more sophisticated version would then look at the organizations capacity, budgets, etc. and determine if they have the right resources to close the gap. 
Potential Questions to answer:
- What would be the impact on the organization if they did not close the gap?
- If not, what would be required to close the gap?
- What remediation reccomendations would you make to the organization?

Let's say that we are looking at the Sales Department. 
We want to get $10M in sales this year. Last year we had $8M in sales. I would like an AI tool that takes a look at a bunch of different things to see what is the likelihood that sales can actually go to $10M. It's going to have to take a look at headcount, budget, projects, problems, and come back with some feedback on where the sales department needs to focus their effort to get their sales to the target level.
If the target level is infeasible, I would like the tool to identify what it thinks the target level should be, or what would need to change to make that target level feasible.

## Simple solution:

Specify a timeframe to do the analysis, lets pick a **year**.
Given a department or business function and key results achieved last year. 
to get increase in X % revenue you need Y % more people, associated backroom support, associated budget, etc.
assume each department has all of it's objectives
1. Calculate the change required to reach the new target.
2. Bring up some objectives from last year and some possible new objectives that could be used to reach the new target.
3. compare the key results from last year to the new target and what the new key results would need to be in order to achieve the new target.
goal is given:
- name: <verb> <outcome>
assume:
- name: increase sales revenue by 20% compared to last year
1. get the things that would help achieve the taget
    1. increase headcount
    2. increase price
    3. enter new markets
    4. introduce new products
    5. combination
    6. increase customer retention / decrease churn
    7. 
2. note which of those things are talked about happening in the future 
3. provide a summary indicating if work is or is not being done to achieve the goal

## Sophisticated solution:

First we need to decide on what the smallest timeframe we want to consider, since each decision/action will take place and performance will be measured in intervals. For example, we could consider the smallest timeframe to be a day, a week, a month, a quarter, a year, etc. I think we should consider a quarter to be the smallest timeframe we consider at this point in time.
I need a tool that takes a look at a department (ex: sales) and understands the following:
- **what** the department achieved over the **last timeframe** (ex: $8M in sales selling products/services x, y, z, to customers a, b, c; over 1 quarter)
- the department's past performance (ex: 10% growth in sales, 5% growth in sales, 2% growth in sales)
- culture (startup vs. mature, vacation season vs. busy season, relaxed vs. stressed, etc.)
- **how** the department achieved what they did last year
    - strategy map
    - objectives
    - key results
    - processes (in place and in development) (self sustaining, manual, automated, etc.)
- the resources that the department has at their disposal (what they are using and what they could be using)
    - understands the department's budget (ex: $1M)
    - understands the department's capacity and what they are assigned to (ex: 10 sales people, 1 sales manager, 1 sales director, 1 sales VP, 1 sales admin, 1 sales operations manager, 1 sales operations analyst)
    - understands the department's projects and timelines
    - understands the department's technology/tools/partnerships/potential deals and the revenue they could bring in
- the department's internal problems/risks
    - understands the department's processes/bottlenecks (ex: process 1, process 2, process 3)
    - understand the department's various costs for increase in capacity or for new technologies or tools
- understands their external problems/risks from their competitors
- understands new potential opportunities they (or their competitors) could take advantage of 
    - and the revenue 
    - and associated probability of success
- different types of revenue generation: active, passive, recurring, one-time
- different types of costs: fixed, variable, one-time, recurring
- different types of capacity: run the business, change the business
Then provide feedback on the following possible actions that could be taken, noting: (+- impact on Revenue (R), +- impact on Costs (C), +- impact on Capacity/Energy (E))
- what if they did nothing and everyone was fired? what remaining processes are in place that drive recurring revenue, how long do those things last? (baseline)
- what if they did the same thing as last year? (what are the delta's in R, C, E from baseline?)
- how could they lower costs or free up resources? (what would be the associated impact on revenue? what could be done with the freed up resources?)
- what if they maximized revenue (what is impact on costs? and capacity?)
- what if they minimized costs (what is impact on revenue? and capacity?)
- what if they maximized capacity (what is impact on revenue? and costs?)
- what if they minimized capacity (what is impact on revenue? and costs?)
Then, provide a range of targets that the department could achieve and the associated changes that may result in those targets being achieved.
- understands the department's _target_ (ex: get $10M in sale's selling products/services x, y, z, to customers a, b, c)
Then, provide feedback on the possibility of the department achieving their initial target.
"""
