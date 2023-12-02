from hermetic.agents.langchain_chat_agent import LangchainChatAgent
from hermetic.agents.openai_chat_agent import OpenAIChatAgent
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage 
from hermetic.core.prompt_mgr import PromptMgr
from hermetic.core.environment import Environment
from hermetic.presenters.gradio_presenter import GradioPresenter


NAME = 'pirate'
MODEL = 'meta-llama/Llama-2-13b-chat-hf'
#MODEL = 'gpt-3.5-turbo'
class Pirate(OpenAIChatAgent):

    def __init__(self, env):
        # Call the super class constructor
        super().__init__(environment = env, id=NAME, model=MODEL)
        env.add_agent(NAME, self)
        self.pm = self.env.prompt_mgr
        
        # bind to the system message
        sys_msg = self.pm.bind('system_msg')

        # Let's add our system message to the message history
        #self.message_history = [SystemMessage(content=sys_msg.render())]
        self.message_history = [{
            'role': 'system',
            'content': sys_msg.render()
        }]

    # Our superclass takes care of the details, all we need to do
    # is to update the message history
    def update_message_history(self, inp): 
        # If we wanted we could add additional details here. 
        #self.message_history.append(HumanMessage(content=inp))
        self.message_history.append({
            'role': 'user',
            'content': inp
        })


# Now let's set up the enviroment and presenter
env = Environment(store = None, prompt_mgr = PromptMgr(hot_reload=True))

# Let's add our agent to the environment
pirate = Pirate(env)
env.set_primary_agent(NAME)

# Now present graphically. 

presenter = GradioPresenter(app_name='Pirate', 
                            env=env)

# This starts the UI. 
presenter.present()



    