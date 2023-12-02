from base.cli_conversation import CliConversation
from connectors.kg_connector import KGConnector
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI


class KGConversation(CliConversation):

    def __init__(self, companion):
        super().__init__(companion)
        
        prompt = PromptTemplate(
            input_variables = self.companion.input_variables(),
            template = self.companion.prompt_template()
        )

        memory = KGConnector().get_client()
        self.conversation_chain = LLMChain(
            llm = ChatOpenAI(
                model_name='gpt-3.5-turbo',
                openai_api_key=self.creds['open_ai_api_key'],
                temperature=0.6
            ),
            prompt = prompt,
            memory=memory,
            verbose=True
        )