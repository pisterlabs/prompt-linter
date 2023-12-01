from langchain.agents.agent import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.schema.messages import SystemMessage

from config import OPENAI_API_KEY
from faq_prompt import FAQ_PROMPT
from utils import custom_google_search_tool

Tools = [
    custom_google_search_tool
]


class OpenAIChatAgent:
    system_message = SystemMessage(
        content="You are a helpful AI assistant."
    )

    extra_prompt_messages = [
        SystemMessagePromptTemplate.from_template(
            """\n---CHAT HISTORY: {chat_history}\n---"""
        )
    ]

    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name='gpt-3.5-turbo-0613',  # gpt 3.5 turbo snapshot with function calling data
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=OpenAIFunctionsAgent.from_llm_and_tools(
                self.llm,
                Tools,
                system_message=self.system_message,
                extra_prompt_messages=self.extra_prompt_messages,
            ),
            tools=Tools,
        )

    def run(self, chat_history, human_input, st_gen):
        st_callback = StreamlitCallbackHandler(st_gen.container())
        chat_history = [f"{message['role']}: {message['content']}" for message in chat_history]

        ai_response = self.agent_executor.run(
            chat_history=str(chat_history),
            input=human_input,
            callbacks=[st_callback],
            verbose=True,
        )

        return ai_response


class GenFAQsLLM:
    def __init__(self, llm_temp: float = 1.0):
        self.llm_temp = llm_temp
        self.faq_prompt_template = FAQ_PROMPT

        self.llm = OpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name='gpt-3.5-turbo-instruct',
            temperature=self.llm_temp,
            max_tokens=64,
        )

        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.faq_prompt_template,
        )

    def run(self, chat_history, human_input, ai_response, n_faqs=4):
        chat_history = [f"{message['role']}: {message['content']}" for message in chat_history]
        input_dict = {
            "chat_history": chat_history,
            "human_input": human_input,
            "ai_response": ai_response,
            "tools": str({tool.name: tool.description for tool in Tools}),
        }
        input_list = [input_dict for _ in range(n_faqs)]

        faqs = self.llm_chain.apply(input_list)

        return [faq['text'] for faq in faqs]
