from openai import OpenAI
import openai
import json
import time

from assistant_functions import *
import assistant_functions

def import_all_from_module(module_name):
    module = importlib.import_module(module_name)
    importlib.reload(module)
    globals().update({k: getattr(module, k) for k in dir(module) if not k.startswith('_')})

def get_all_tools():
    module = importlib.import_module('assistant_functions')
    importlib.reload(module)
    return [getattr(module, k) for k in dir(module) if k.endswith('tool')]

def color(text, color):
    if color == 'green':
        return "\033[1;32;40m" + text + "\033[0m"
    elif color == 'red':
        return "\033[1;31;40m" + text + "\033[0m"
    elif color == 'blue':
        return "\033[1;34;40m" + text + "\033[0m"
    elif color == 'yellow':
        return "\033[1;33;40m" + text + "\033[0m"
    elif color == 'magenta':
        return "\033[1;35;40m" + text + "\033[0m"
    elif color == 'cyan':
        return "\033[1;36;40m" + text + "\033[0m"
    elif color == 'white':
        return "\033[1;37;40m" + text + "\033[0m"
    else:
        return text

class AssistantRun:
    def __init__(self, client, name, description, model, tools, file_ids):
        self.client = client
        self.name = name
        self.description = description
        self.model = model
        self.tools = tools
        self.file_ids = file_ids
        self.assistant = client.beta.assistants.create(
            name=name,
            description=description,
            model=model,
            tools=tools,
            file_ids=file_ids
        )
        self.thread = client.beta.threads.create()

    def update(self, tools=None, file_ids=None):
        if tools is None:
            tools = self.tools
        if file_ids is None:
            file_ids = self.file_ids
        self.assistant = self.client.beta.assistants.update(
            self.assistant.id,
            tools=tools,
            file_ids=file_ids
        )
    
    def perform_action(self, action, tool_outputs):
        try:
            args = json.loads(action.function.arguments)
            import_all_from_module("assistant_functions")
            func = eval(action.function.name)
            if func == write_new_function:
                args["assistant_run"] = self
            output = func(**args)
        except NameError:
            output = "Action not found."
        except Exception as e:
            output = f"Error: {e}"
        
        tool_outputs.append(
            {
                "tool_call_id": action.id,
                "output": output,
            }
        )
    
    def run(self, user_input):
        thread_message = self.client.beta.threads.messages.create(
            self.thread.id,
            role="user",
            content=user_input,
        )
        run = self.client.beta.threads.runs.create(thread_id=self.thread.id, assistant_id=self.assistant.id)

        # Polling the run status
        while True:
            run = self.client.beta.threads.runs.retrieve(run_id=run.id, thread_id=run.thread_id)
            status = run.status
            if status == 'completed':
                break
            elif status == 'failed':
                print("Run failed")
                break
            elif status == 'requires_action':
                actions = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for action in actions:
                    self.perform_action(action, tool_outputs)
                run = self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
            time.sleep(1)

        thread_messages = self.client.beta.threads.messages.list(run.thread_id)
        last_message = thread_messages.data[0]
        print(color("Assistant: ", 'red') + last_message.content[0].text.value)
    


if __name__ == "__main__":

    get_all_tools()
    
    client = OpenAI(api_key=open("../key.txt").read().strip())

    assistant = AssistantRun(
        client,
        name="Personal Assistant",
        description="You are a personal assistant that helps people with their day-to-day tasks. \
                You are able to read and write to files, and also create new files. \
                You can also write new functions and use them as new tools. \
                Show initiative and help the user with their tasks.",
        model="gpt-4-1106-preview",
        tools = [
            {"type": "code_interpreter"},
            {"type": "retrieval"}] +
            get_all_tools(),
        file_ids=[]
    )

    while True:
        # print You: in color
        user_input = input(color("You: ", 'green'))
        if user_input == "quit":
            break
        assistant.run(user_input)
