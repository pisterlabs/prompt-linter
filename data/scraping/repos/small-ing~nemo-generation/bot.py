from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action


# load the .env file
load_dotenv()

class Bot:
    def __init__(self, config_path: str = "config/base"):
        '''
        The constructor method to create the Bot class, 
        and instantiate the LLM and associated guardrails

        ### Parameters:
        - `config_path`: The path to the config file to use for the bot
        '''

        self.rails = LLMRails(RailsConfig.from_path(config_path), verbose=True)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

    @action()
    async def chat(self, history: []):
        # process the message through the LLM and guardrails
        rail = await self.rails.generate_async(messages=history)
        return rail

    def load_actions(self):
        '''
        Registering the actions to reference in Colang
        * self.rails.register_action(self.chat, name="chat")
        
        where the first argument is the name of any function with the @action() decorator
        and the second argument is the name of the action in Colang
        '''

        self.rails.register_action(self.chat, name="chat")