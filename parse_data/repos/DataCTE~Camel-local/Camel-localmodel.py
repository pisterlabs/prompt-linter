import os
import datetime

from typing import List
from langchain.callbacks import get_openai_callback
from langchain.llms import GPT4All
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
import time
from threading import Thread
import json
from typing import List
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Define the model file path here
model_file_path = "/home/dev-1/.local/share/nomic.ai/GPT4All/ggml-gpt4all-j-v1.3-groovy.bin"

class CAMELAgent:
    def __init__(self, model_file_path: str, max_memory_size: int = 10) -> None:
        template = """Question: {question}

        Answer: Let's think step by step."""

        self.prompt = PromptTemplate(template=template, input_variables=["question"])
        self.callbacks = [StreamingStdOutCallbackHandler()]
        self.llm = GPT4All(model=model_file_path)
        self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)
        self.memory = []
        self.max_memory_size = max_memory_size

    def reset(self) -> None:
        self.memory = []

    def _run_with_timeout(self, func, args=(), timeout=60):
        result = [None]

        def target():
            result[0] = func(*args)

        thread = Thread(target=target)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            thread.join()
            raise TimeoutError("Model response timeout")

        return result[0]

    def step(self, role: str, message: str) -> str:
        input_with_memory = '\n'.join([json.dumps(msg) for msg in self.memory] + [f"{role}: {message}"])

        try:
            response = self._run_with_timeout(self.llm_chain.run, args=(input_with_memory,), timeout=800)
        except TimeoutError:
            response = "Model response timed out"

        # Update memory with current input and response
        self.memory.append({
            'role': role,
            'message': message,
            'response': response
        })

        # Truncate the memory if it exceeds the maximum size
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size:]

        return response

assistant_role_name = "Game Dev Coder"
user_role_name = "Game Enthusiast"
task = "Design a text adventure game set in Singapore"

word_limit = 50  # word limit for task brainstorming

task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
task_specifier_prompt = (
    """Here is a task that {assistant_role_name} will discuss with {user_role_name} to: {task}.
Please make it more specific. Be creative and imaginative.
Please reply with the full task in {word_limit} words or less. Do not add anything else."""
)
task_specifier_template = SystemMessagePromptTemplate.from_template(template=task_specifier_prompt)
task_specify_agent = CAMELAgent(model_file_path)
task_specifier_msg = task_specifier_template.format_messages(
    assistant_role_name=assistant_role_name,
    user_role_name=user_role_name,
    task=task,
    word_limit=word_limit
)[0]
specified_task_msg = task_specify_agent.step("user", task_specifier_msg.content)
specified_task = specified_task_msg
print(f"Specified task: {specified_task}")


assistant_inception_prompt = (
    """Never forget you are a {assistant_role_name} and I am a {user_role_name}. Never flip roles!
We share a common interest in collaborating to successfully complete a task.
You must help me to complete the task.
Here is the task: {task}. Never forget our task!
I will instruct you based on your expertise and my needs to complete the task.

I must give you one question at a time.
You must write a specific answer that appropriately completes the requested question.
You must decline my question honestly if you cannot comply the question due to physical, moral, legal reasons or your capability and explain the reasons.
Do not add anything else other than your answer to my instruction.

Unless I say the task is completed, you should always start with:

My response: <YOUR_SOLUTION>

<YOUR_SOLUTION> should be specific and descriptive.
Always end <YOUR_SOLUTION> with: Next question."""
)

user_inception_prompt = (
    """Never forget you are a {user_role_name} and I am a {assistant_role_name}. Never flip roles! You will always ask me.
We share a common interest in collaborating to successfully complete a task.
I must help you to answer the questions.
Here is the task: {task}. Never forget our task!
You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways:

1. Instruct with a necessary input:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>

2. Instruct without any input:
Instruction: <YOUR_INSTRUCTION>
Input: None

The "Instruction" describes a task or question. The paired "Input" provides further context or information for the requested "Instruction".

You must give me one instruction at a time.
I must write a response that appropriately completes the requested instruction.
I must decline your instruction honestly if I cannot perform the instruction due to physical, moral, legal reasons or my capability and explain the reasons.
You should instruct me not ask me questions.
Now you must start to instruct me using the two ways described above.
Do not add anything else other than your instruction and the optional corresponding input!
Keep giving me instructions and necessary inputs until you think the task is completed.
When the task is completed, you must only reply with a single word <TASK_DONE>.
Never say <TASK_DONE> unless my responses have solved your task."""
)


def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task
    )[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task
    )[0]

    return assistant_sys_msg, user_sys_msg


assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task)
assistant_agent = CAMELAgent(model_file_path)
user_agent = CAMELAgent(model_file_path)
# Reset agents
assistant_agent.reset()
user_agent.reset()

# Initialize chats
assistant_msg = HumanMessage(
    content=(f"{user_sys_msg.content}. "
             "Now start to give me introductions one by one. "
             "Only reply with Instruction and Input."))

user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
user_msg = assistant_agent.step("user", user_msg.content)


def write_conversation_to_file(conversation: List[str], filename: str) -> None:
    """
    Write a conversation to a text file with a timestamp in its filename.

    Parameters:
    - conversation (List[str]): A list of strings representing the conversation turns.
    - filename (str): The name of the file to write the conversation to.

    Returns:
    None
    """
    def timestamp() -> str:
        """
        Convert the current date and time into a custom timestamp format.

        Returns:
        str: The current date and time in the format HHMMDDMMYYYY.
        """
        now = datetime.datetime.now()
        timestamp = now.strftime("%H%M%d%m%Y")
        return timestamp

    def append_timestamp_to_filename(filename: str) -> str:
        """
        Append a timestamp to a filename before the extension.

        Parameters:
        - filename (str): The original filename.

        Returns:
        str: The filename with a timestamp appended.
        """
        base, extension = os.path.splitext(filename)
        new_filename = f"{base}-{timestamp()}{extension}"
        return new_filename

    filename = append_timestamp_to_filename(filename)

    with open(filename, 'w') as f:
        for turn in conversation:
            f.write(f"{turn}\n\n")


print(f"Original task prompt:\n{task}\n")
print(f"Specified task prompt:\n{specified_task}\n")

conversation = []
chat_turn_limit, n = 15, 0
while n < chat_turn_limit:
    n += 1
    user_ai_msg = user_agent.step("assistant", assistant_msg.content)
    user_msg = HumanMessage(content=user_ai_msg)
    print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")
    conversation.append(f"AI User ({user_role_name}):\n\n{user_msg.content}")

    assistant_ai_msg = assistant_agent.step("user", user_msg.content)
    assistant_msg = HumanMessage(content=assistant_ai_msg)
    print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
    conversation.append(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}")

    if "<TASK_DONE>" in user_msg.content:
        break

print(f"Total Successful Requests: {assistant_agent.llm_chain.llm.successful_requests}")
print(f"Total Tokens Used: {assistant_agent.llm_chain.llm.total_tokens}")
print(f"Prompt Tokens: {assistant_agent.llm_chain.prompt_tokens}")
print(f"Completion Tokens: {assistant_agent.llm_chain.completion_tokens}")
print(f"Total Cost (USD): ${assistant_agent.llm_chain.total_cost}")

write_conversation_to_file(conversation, 'conversation.txt')
