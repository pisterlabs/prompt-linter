from todoist_api_python.api import TodoistAPI
import openai
from dotenv import load_dotenv
import os
from datetime import datetime

# load environment variables from .env file
load_dotenv()

# get API keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
todoist_api_key = os.getenv("TODOIST_API_KEY")

# set up Todoist API client
todoist_api = TodoistAPI(todoist_api_key)

# generate a todo using ChatGPT
def generate_todo():
    today = datetime.today().strftime('%Y-%m-%d')

    # prompt the user to enter a goal
    goal = input("Enter your goal: ")

    # prompt the user to enter a due date for the task
    prompt = f"Today's date: {today}. I want you to act as a to-do list generator. Generate a SMART task that aligns with my {goal}. What is one thing I should add to my to-do list to achieve this goal? Your task should be specific, measurable, achievable, relevant, and time-bound. Respond in one line and include a human-defined task due date."
    try:
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=50)
        todo = response.choices[0].text.strip()
    except Exception as error:
        print(error)
        todo = None
    return todo

# add the generated todo to Todoist with a due date
def add_todo_to_todoist():
    todo = generate_todo()
    if todo:
        try:
            # print the generated task
            print(f"Generated task: '{todo}'")
            
            # prompt the user to enter a due date for the task
            due_string = input("Enter a due date for the task (e.g. 'next Monday', 'tomorrow'): ")
            task = todoist_api.add_task(content=todo, project_id=None, due_string=due_string)
            print(f"Task '{todo}' with due date '{due_string}' added to Todoist!")
        except Exception as error:
            print(error)

# call the add_todo_to_todoist function to add a new task to Todoist
add_todo_to_todoist()
