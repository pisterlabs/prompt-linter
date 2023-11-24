import environment
import os

# Anthropic
def AnthropicChatModel():
    from langchain.chat_models import ChatAnthropic
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    chat = ChatAnthropic(anthropic_api_key=os.environ.get['ANTHROPIC_API_KEY'])
    ## ChatAnthropic also supports async and streaming functionality
    # from langchain.callbacks.manager import CallbackManager
    # from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    # await chat.agenerate([messages])
    # chat = ChatAnthropic(streaming=True, verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    # chat(messages)
    return chat

def PromptLayerChatModel():
    # pip install promptlayer
    import os
    import promptlayer
    from langchain.chat_models import PromptLayerChatOpenAI
    from langchain.schema import HumanMessage
    # os.environ["PROMPTLAYER_API_KEY"] = "**********"
    chat = PromptLayerChatOpenAI(pl_tags=["langchain"])
    chat([HumanMessage(content="I am a cat and I want")])
    chat = PromptLayerChatOpenAI(return_pl_id=True)
    chat_results = chat.generate([[HumanMessage(content="I am a cat and I want")]])

    for res in chat_results.generations:
        pl_request_id = res[0].generation_info["pl_request_id"]
        promptlayer.track.score(request_id=pl_request_id, score=100)

    return chat

def AzureChatModel():
    from langchain.chat_models import AzureChatOpenAI
    from langchain.schema import HumanMessage
    BASE_URL = "https://${TODO}.openai.azure.com"
    API_KEY = "..."
    DEPLOYMENT_NAME = "chat"
    chat = AzureChatOpenAI(
        openai_api_base=BASE_URL,
        openai_api_version="2023-03-15-preview",
        deployment_name=DEPLOYMENT_NAME,
        openai_api_key=API_KEY,
        openai_api_type = "azure",
    )

def OpenAIChatModel():
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    chat = ChatOpenAI(temperature=0)
    return chat

def defaultChatModel():
    # chatModel = AnthropicChatModel()
    chatModel = PromptLayerChatModel()
    return chatModel


from langchain.schema import HumanMessage, SystemMessage
messages = [
    HumanMessage(content="Translate this sentence from English to French. I love programming.")
]
# messages = [
#     SystemMessage(content="You are a helpful assistant that translates English to French."),
#     HumanMessage(content="Translate this sentence from English to French. I love programming.")
# ]

chat = defaultChatModel()
chat(messages)

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# Or If you wanted to construct the MessagePromptTemplate more directly, you could create a PromptTemplate outside and then pass it in, eg:
prompt=PromptTemplate(
    template="You are a helpful assistant that translates {input_language} to {output_language}.",
    input_variables=["input_language", "output_language"],
)
system_message_prompt_2 = SystemMessagePromptTemplate(prompt=prompt)
assert system_message_prompt == system_message_prompt_2

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# As string
output = chat_prompt.format(input_language="English", output_language="French", text="I love programming.")
# or alternatively 
output_2 = chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_string()
assert output == output_2

# As ChatPromptValue
chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.")

# As list of Message objects
chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages()

# get a chat completion from the formatted messages
chat(chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages())


from langchain.prompts import ChatMessagePromptTemplate

prompt = "May the {subject} be with you"

chat_message_prompt = ChatMessagePromptTemplate.from_template(role="Jedi", template=prompt)
chat_message_prompt.format(subject="force")

from langchain.prompts import MessagesPlaceholder

human_prompt = "Summarize our conversation so far in {word_count} words."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)


chat_prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="conversation"), human_message_template])
human_message = HumanMessage(content="What is the best way to learn programming?")
ai_message = AIMessage(content="""\
1. Choose a programming language: Decide on a programming language that you want to learn. 

2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.

3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
""")

chat_prompt.format_prompt(conversation=[human_message, ai_message], word_count="10").to_messages()


