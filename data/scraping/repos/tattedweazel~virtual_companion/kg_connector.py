from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationKGMemory
from tools.secret_squirrel import SecretSquirrel


class KGConnector():

    def __init__(self):
        self.creds = SecretSquirrel().stash
        self.llm = ChatOpenAI(
                model_name='gpt-3.5-turbo',
                openai_api_key=self.creds['open_ai_api_key'],
                temperature=0
            )


    def get_client(self):
        return ConversationKGMemory(llm=self.llm)