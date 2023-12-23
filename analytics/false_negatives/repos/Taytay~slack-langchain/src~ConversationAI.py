import asyncio
from langchain import ConversationChain
from langchain.agents import Agent, Tool, initialize_agent
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)
from langchain.utilities import GoogleSerperAPIWrapper, SerpAPIWrapper
from .conversation_utils import is_asking_for_smart_mode, get_recommended_temperature

from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from slack_sdk import WebClient
# How to do a search over docs with conversation:
#https://langchain.readthedocs.io/en/latest/modules/memory/examples/adding_memory_chain_multiple_inputs.html
# People talking about the parsing error: https://github.com/hwchase17/langchain/issues/1657

from .AsyncStreamingSlackCallbackHandler import AsyncStreamingSlackCallbackHandler


DEFAULT_MODEL="gpt-3.5-turbo"
UPGRADE_MODEL="gpt-4"
DEFAULT_TEMPERATURE=0.3

class ConversationAI:
    def __init__(
        self, bot_name:str, slack_client:WebClient, existing_thread_history=None, model_name:str=None
    ):
        self.bot_name = bot_name
        self.existing_thread_history = existing_thread_history
        self.model_name = None
        self.agent = None
        self.model_temperature = None
        self.slack_client = slack_client
        self.lock = asyncio.Lock()

    async def create_agent(self, sender_user_info, initial_message):
        print(f"Creating new ConversationAI for {self.bot_name}")

        sender_profile = sender_user_info["profile"]

        # TODO: If we are picking up from where a previous thread left off, we shouldn't be looking at the initial message the same way, and should use the original message as the "initial message"

        # Call both async methods simultaneously
        smart_mode_task = asyncio.create_task(is_asking_for_smart_mode(initial_message))
        recommended_temperature_task = asyncio.create_task(get_recommended_temperature(initial_message, DEFAULT_TEMPERATURE))

        # Await the results
        requested_smart_mode, recommended_temperature = await asyncio.gather(smart_mode_task, recommended_temperature_task)

        if requested_smart_mode:
            self.model_name = UPGRADE_MODEL
        else:
            self.model_name = DEFAULT_MODEL

        if recommended_temperature is not None:
            recommended_temperature = max(0.0, recommended_temperature)
            recommended_temperature = min(1.0, recommended_temperature)
        else:
            recommended_temperature = DEFAULT_TEMPERATURE

        self.model_temperature = recommended_temperature
        
        print("Will use model: " + self.model_name)
        print(f"Will use temperature: {self.model_temperature}")

        model_facts = f"You are based on the OpenAI model {self.model_name}. Your 'creativity temperature' is set to {self.model_temperature}."

        #additional_kwargs={"name": "example_user"}
        prompt = ChatPromptTemplate.from_messages(
            [
                # TODO: We need a way to label who the humans are - does the HumanMessagePromptTemplate support this?
                # TODO: Extract this prompt out of this file
                SystemMessagePromptTemplate.from_template(
                    f"""The following is a Slack chat thread between users and you, a Slack bot named {self.bot_name}.
You are funny and smart, and you are here to help.
If you are not confident in your answer, you say so, because you know that is helpful.
You don't have realtime access to the internet, so if asked for information about a URL or site, you should first acknowledge that your knowledge is limted before responding with what you do know.
Since you are responding in Slack, you format your messages in Slack markdown, and you LOVE to use Slack emojis to convey emotion.
If the human appears to be talking to someone else, especially if they start their message with addressing someone else like "@not-the-bot-name", or they am about you in the 3rd person, you will ONLY respond with the emoji: ":speak_no_evil:"
Some facts about you:
{model_facts}
"""
                ),
                HumanMessagePromptTemplate.from_template(f"""Here is some information about me. Do not respond to this directly, but feel free to incorporate it into your responses:
I'm  {sender_profile.get("real_name")}. 
Since we're talking in Slack, you can @mention me like this: "<@{sender_user_info.get("id")}>"
My title is: {sender_profile.get("title")}
My current status: "{sender_profile.get("status_emoji")}{sender_profile.get("status_text")}"
Please try to "tone-match" me: If I use emojis, please use lots of emojis. If I appear business-like, please seem business-like in your responses. Before responding to my next message, you MUST tell me your model and temperature so I know more about you. Don't reference anything I just asked you directly."""),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ]
        )
        self.callbackHandler = AsyncStreamingSlackCallbackHandler(self.slack_client)

        llm = ChatOpenAI(model_name = self.model_name, temperature=self.model_temperature, request_timeout=60, max_retries=3, streaming=True, verbose=True, callback_manager=AsyncCallbackManager([self.callbackHandler]))
        # This buffer memory can be set to an arbitrary buffer
        memory = ConversationBufferMemory(return_messages=True)

        # existing_thread_history is an array of objects like this:
        # {
        #     "taylor": "Hello, how are you?",
        #     "bot": "I am fine, thank you. How are you?"
        #     "kevin": "@taylor, I'm talking to you now"
        #     "taylor": "@kevin, Oh cool!"
        # }
        # We should iterate through this and add each of these to the memory:
        # Unfortunately, we can't use the human's name because the memory doesn't seem to support that yet
        existing_thread_history = self.existing_thread_history
        if existing_thread_history is not None:
            for message in existing_thread_history:
                sender_name = list(message.keys())[0] # get the first key which is the name (assuming only one key per dictionary)
                message_content = list(message.values())[0] # get the first value which is the message content
                if sender_name == "bot":
                    memory.chat_memory.add_ai_message(message_content)
                else:
                    memory.chat_memory.add_user_message(message_content)

        self.memory = memory
        self.agent = ConversationChain(
            memory=memory, prompt=prompt, llm=llm, verbose=True
        )
        return self.agent

    async def get_or_create_agent(self, sender_user_info, message):
        if self.agent is None:
            self.agent = await self.create_agent(sender_user_info, message)
        return self.agent

    async def respond(self, sender_user_info, channel_id:str, thread_ts:str, message_being_responded_to_ts:str, message:str):
        async with self.lock:
          agent = await self.get_or_create_agent(sender_user_info, message)
          # TODO: This is messy and needs to be refactored...
          print("Starting response...")
          await self.callbackHandler.start_new_response(channel_id, thread_ts)
          # Now that we have a handler set up, just telling it to predict is sufficient to get it to start streaming the message response...
          response = await self.agent.apredict(input=message)
          return response
        
def get_conversational_agent(model_name="gpt-3.5-turbo"):
  search = GoogleSerperAPIWrapper()
  # TODO: File a PR to fix this return_messages=True thing
  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
  search = SerpAPIWrapper()
  tools = [
      Tool(
          name = "search",
          func=search.run,
          description="(ONLY if your confidence in your answer is below 0.2, use this tool to search for information)"
      ),
  ]
  llm=ChatOpenAI(temperature=0, model=model_name, verbose=True, request_timeout=30, max_retries=0)
  agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True, memory=memory, request_timeout=30)
  return agent_chain




# class CustomConversationAgent(Agent):
#     def __init__(self, llm_chain, allowed_tools=None):
#         super().__init__(llm_chain, allowed_tools)

#     def parse_output(self, output):
#         lines = output.strip().split("\n")
#         is_talking_to_ai = "no"
#         switch_to_smarter_mode = "no"

#         for line in lines:
#             if line.startswith("Is_talking_to_AI:"):
#                 is_talking_to_ai = line.split(":")[1].strip()
#             elif line.startswith("Switch_to_smarter_mode:"):
#                 switch_to_smarter_mode = line.split(":")[1].strip()

#         return is_talking_to_ai, switch_to_smarter_mode
