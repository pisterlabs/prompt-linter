import os
import textwrap

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter
from pygments.lexers.data import JsonLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.special import TextLexer

# Initial Setup.
load_dotenv()
chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
chat.model_name = "gpt-4"


# Helpers
# Create a TerminalFormatter with line wrapping enabled
def text_format(text_object):
    # Convert the ChatPromptTemplate object to a string representation
    formatted_prompt = str(text_object)

    # Wrap the lines of the formatted prompt
    wrapped_prompt = textwrap.indent(textwrap.fill(formatted_prompt, width=80),
                                     "    ")

    # Highlight the wrapped prompt using the TextLexer
    formatted_output = highlight(wrapped_prompt, TextLexer(),
                                 TerminalFormatter())

    # Return the formatted output
    return formatted_output

##########################  Basic Usage  ###############################

# # Basic old school way
# simple_message = [HumanMessage(
#     content="Give me a python script that queries the Google Ads API "
#             "for top performing ad extensions")]
#
# response = chat(simple_message)
# print(response)


# # Chat message
# messages = [
#     SystemMessage(content="Say the opposite of what the user says"),
#     HumanMessage(content="I love Ireland.")
# ]
# print(chat(messages))


# # Chat message with history
# messages = [
#     SystemMessage(content="Say the opposite of what the user says"),
#     HumanMessage(content="I love Ireland."),
#     AIMessage(content="I hate Ireland."),
#     # HumanMessage(content="The moon is dull"),
#     # AIMessage(content='The moon is exciting.'),
#     HumanMessage(content="What was the first thing I said?"),
# ]
# print(chat(messages))


# Batch messages


# batch_messages = [
#     [
#         SystemMessage(
#             content="You are a helpful word machine that creates an  "
#                     "alliteration for a spooky minecraft story using "
#                     "a base word"
#         ),
#         HumanMessage(content="Base word: Dave"),
#     ],
#     [
#         SystemMessage(
#             content="You are a helpful word machine that creates an "
#                     "alliteration for a spooky minecraft story using "
#                     "a base word"
#         ),
#         HumanMessage(content="Base word: Charlie"),
#     ],
# ]
# print(chat.generate(batch_messages))


#########################  Prompt Usage  ##############################

# With one or more MessagePromptTemplates you can build a ChatPromptTemplate
# Make SystemMessagePromptTemplate
prompt = PromptTemplate(
    template="Propose a creative funny minecraft character given a name: "
             "{seed_name}, and a favorite food: {seed_food}",
    input_variables=["seed_name", "seed_food"]
)

system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)

# Output of system_message_prompt
print(text_format(system_message_prompt.format(seed_name="Dave",
                                               seed_food="Egg on toast")))

# Make HumanMessagePromptTemplate
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Create ChatPromptTemplate: Combine System + Human
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])
print(text_format(chat_prompt))

# Full chat Prompt composition.
chat_prompt_with_values = chat_prompt.format_prompt(seed_name="Dave",
                                                    seed_food="Egg on toast",
                                                    text="I'm from Ireland and "
                                                         "love sweets & lego'.")

# See how it looks.
print(text_format(chat_prompt_with_values.to_messages()))

# # Sent it to GPT
# response = chat(chat_prompt_with_values.to_messages()).content
# print(text_format(response))


# With Streaming
chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
chat.model_name = "gpt-4"
resp = chat(chat_prompt_with_values.to_messages())
print(text_format(resp))

