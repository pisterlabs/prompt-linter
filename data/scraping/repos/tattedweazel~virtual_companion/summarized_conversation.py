from base.cli_conversation import CliConversation
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory


class SummarizedConversation(CliConversation):

    def __init__(self, companion):
        super().__init__(companion)
        
        prompt = PromptTemplate(
            input_variables = self.companion.input_variables(),
            template = self.companion.prompt_template()
        )

        self.conversation_chain = LLMChain(
            llm = ChatOpenAI(
                model_name='gpt-3.5-turbo',
                openai_api_key=self.creds['open_ai_api_key'],
                temperature=0.5
            ),
            prompt = prompt,
            memory=ConversationSummaryMemory(
                llm=ChatOpenAI(
                    model_name='gpt-3.5-turbo',
                    openai_api_key=self.creds['open_ai_api_key'],
                    temperature=0.2
                ),
                ai_prefix=self.companion.name,
                human_prefix=self.companion.human_name
            ),
            verbose=True
        )


    