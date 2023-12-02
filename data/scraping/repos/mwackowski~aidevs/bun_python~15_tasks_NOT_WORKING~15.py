import asyncio

from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from helper import currentDate, parseFunctionCall, rephrase
from todoist import addTasks, closeTasks, listUncompleted, updateTasks
from schema import addTasksSchema, finishTasksSchema, getTasksSchema, updateTasksSchema

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model = ChatOpenAI(model="gpt-4-0613").bind(
    functions=[getTasksSchema, addTasksSchema, finishTasksSchema, updateTasksSchema]
)
tools = {
    "getTasks": listUncompleted,
    "addTasks": addTasks,
    "closeTasks": closeTasks,
    "updateTasks": updateTasks,
}


async def act(query):
    print("User: ", query)
    tasks = await listUncompleted()
    conversation = await model.invoke(
        [
            SystemMessage(
                content=f"""
            Fact: Today is {currentDate()}
            Current tasks: ###{', '.join([task.content + ' (ID: ' + str(task.idx) + ')' for task in tasks])}###"""
            ),
            HumanMessage(content=query),
        ]
    )
    action = parseFunctionCall(conversation)
    response = ""
    if action:
        print(f"action: {action['name']}")
        response = await tools[action["name"]](action["args"]["tasks"])
        response = await rephrase(response, query)
    else:
        response = conversation.content
    print(f"AI: {response}\n")
    return response


async def main():
    await act("I need to write a newsletter about gpt-4 on Monday, can you add it?")
    await act("Need to buy milk, add it to my tasks")
    await act("Ouh I forgot! Beside milk I need to buy sugar. Update my tasks please.")
    await act("Get my tasks again.")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
