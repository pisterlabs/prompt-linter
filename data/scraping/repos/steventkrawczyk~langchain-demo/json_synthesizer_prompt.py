from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AIMessage
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage

system_template = """You are a data scientists tasked with creating a test set for evaluating new large language models.
You develop test data sets based on descriptions of specific tasks that the model is expected to perform.
Given a description of a task, and an example of input for the task, create test data for testing the langauge model.

You should format test data as a JSON list of strings, e.g. 
```
{{
    "inputs": ["$YOUR_FIRST_INPUT", "$YOUR_FIRST_INPUT", ...]
}}
```

Everything between the ``` must be valid json. Use different examples from the what the user provides. 
Generate the number of data points specified by the user.
"""
human_message_prompt_template = """Please create data points, in the specified JSON format, for the following task description:
----------------
{text}"""

EXAMPLE_INPUT = """The goal of this language model is to process information about Amazon products.
The input will contain a name, author, and ASIN, in the following JSON format:

```
{{
    "name": "The Art Of War",
    "author": "Sun Tzu",
    "ASIN": "B08Q68NYL2",
}}
```
Your inputs should follow this JSON format.
Everything between the ``` must be valid json.
Generate two inputs.
"""

EXAMPLE_OUTPUT = """{
    "inputs": [{"name": "Meditatons", "author": "Marcus Aurelius", "ASIN": "0812968255"},
               {"name": "The 48 Laws of Power", "author": "Robert Greene", "ASIN": "0226026752"}]
}
"""


CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessage(content=EXAMPLE_INPUT),
        AIMessage(content=EXAMPLE_OUTPUT),
        HumanMessagePromptTemplate.from_template(human_message_prompt_template),
    ]
)