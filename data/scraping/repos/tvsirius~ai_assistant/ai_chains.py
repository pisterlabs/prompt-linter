from langchain.chat_models import ChatOpenAI
from langchain.schema import messages_from_dict, messages_to_dict
from langchain import OpenAI, PromptTemplate, LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory, ConversationSummaryBufferMemory

from ai_roles import ai_init_string

from dotenv import dotenv_values

env_vars = dotenv_values('.env')
OPENAI_API_KEY = env_vars['OPENAI_API_KEY']


def history_formatter(message_list: list[dict]) -> str:
    result = ''
    if message_list and len(message_list) > 0:
        for message in message_list:
            result += message["type"] + ': ' + message["data"]["content"] + "\n"
    return result




template = """{ai_init_string}
{history}
Human: {human_input}
Assistant:"""


class CustomPromt(PromptTemplate):
    def format(self, **kwargs) -> str:
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        kwargs["history"] = history_formatter(messages_to_dict(kwargs["history"]))
        return super().format(**kwargs)


class MyConversation(LLMChain):
    def __init__(self, ai_role, max_token_limit=2000):
        myprompt = CustomPromt(
            input_variables=["history", "human_input"],
            template=template,
            partial_variables={"ai_init_string": ai_init_string[ai_role]['prompt_intro']}
        )
        mychat = ChatOpenAI(temperature=ai_init_string[ai_role]['temperature'], model_name="gpt-3.5-turbo",
                            openai_api_key=OPENAI_API_KEY, )
        super().__init__(
            llm=mychat,
            verbose=True,
            memory=ConversationSummaryBufferMemory(llm=mychat, max_token_limit=max_token_limit, return_messages=True),
            prompt=myprompt,
        )
        # self.ai_role = ai_role
        # self.mychat = mychat
