
import openai
import logging
import re
from injector import Injector

from src.container_config import container
from src.config_manager import ConfigManager  

from src.response_handler import FileResponseHandler
#from src.source_code_response_handler import SourceCodeResponseHandler
from src.agent_manager import AgentManager
from src.message_processors.message_preProcess import MessagePreProcess
from src.presets.preset_handler import PresetHandler
from src.api_helpers import ask_question, get_completion
from src.context.context_manager import ContextManager
from src.context.context import Context

from src.completion.completion_manager import CompletionManager
from src.completion.completion_store import CompletionStore
# from src.completion.completion import Completion
from src.completion.message import Message
from src.prompt_builders.prompt_builder_interface import PromptBuilderInterface



class PromptBuilder(PromptBuilderInterface):
    """
    The PromptBuilder class is responsible for constructing prompts that will be used to generate responses in a conversational system.
    It interacts with various components of the system, such as agents, configuration settings, and completion managers, to build these prompts.

    Attributes:
    agent_manager (AgentManager): An instance of AgentManager, used for getting agent details.
    config (ConfigManager): An instance of ConfigManager, used for getting configuration settings.
    seed_conversations (list): A list used to store seed conversations.
    handler (FileResponseHandler): An instance of FileResponseHandler, used for handling responses.
    preset_handler (PresetHandler): An instance of PresetHandler, used for processing preset prompt values.
    """

    def __init__(self ):
        # self.name = name

        self.agent_manager = container.get(AgentManager)
        self.config = container.get(ConfigManager) 
        self.seed_conversations = []
        self.handler = container.get(FileResponseHandler)   
        #self.preprocessor = container.get(MessagePreProcess)
        self.preset_handler = PresetHandler()

    def build_prompt(self, content_text:str, conversationId:str, agent_name, account_name, context_type="none", max_prompt_chars=6000, max_prompt_conversations=20, context_name=''):
        """
        This method is used to construct a prompt that can be used to generate a response in the conversational system.
        It retrieves and prepares various pieces of information, such as agent details, account details, relevant prompts, and seed prompts.
        It also uses various strategies (e.g., keyword matching, semantic matching) to select relevant prompts based on the given context.

        Parameters:
        content_text (str): The content text that will be part of the constructed prompt.
        conversationId (str): The identifier of the conversation.
        agent_name (str): The name of the agent.
        account_name (str): The name of the account.
        context_type (str): The type of context (e.g.,'keyword', 'semantic'). Default is 'none'.
        completion_merge_type (str): The type of completion merge simple prompt adds completions to begining of content_text
        max_prompt_chars (int): The maximum number of characters allowed in the prompt. Default is 6000.
        max_prompt_conversations (int): The maximum number of conversations allowed in the prompt. Default is 20.

        Returns:
        list: A list of dictionaries that represents the constructed prompt.
        """
        logging.info(f'build_prompt: {context_type}')

        seed_name = self.config.get("seed_name")
        agent = self.agent_manager.get_agent(agent_name)
        completion_manager_store = container.get(CompletionStore)
        account_completion_manager = completion_manager_store.get_completion_manager(agent_name, account_name, agent['language_code'][:2])
        agent_account = self.config.get("agent_internal_account_name")
        agent_completion_manager = completion_manager_store.get_completion_manager(agent_name, agent_account, agent['language_code'][:2])

        # agent properties that are used with the completions API
        model = agent["model"]
        temperature = agent["temperature"]
        num_past_conversations = agent["num_past_conversations"]
        num_relevant_conversations = agent["num_relevant_conversations"]
        use_prompt_reduction = agent["use_prompt_reduction"]
        if "roles_used_in_context" in agent:
            roles_used_in_context = agent["roles_used_in_context"]
        else:
            roles_used_in_context = None

        if "completion_merge_type" in agent:
            completion_merge_type = agent["completion_merge_type"]
        else:
            completion_merge_type = "none"

            

        
        # get the agents seed prompts - this is fixed information for the agent
        agent_roles = ['system']
        
        # we always want to get the latest seed prompts
        agent_matched_seed_ids = self.get_matched_ids(agent_completion_manager, "keyword_match_all", seed_name, 2, num_past_conversations)
        agent_matched_ids = self.get_matched_ids(agent_completion_manager, "keyword", content_text, num_relevant_conversations, num_past_conversations)
        agent_all_matched_ids = agent_matched_seed_ids + agent_matched_ids
        matched_messages_agent = agent_completion_manager.get_completion_messages(agent_all_matched_ids, agent_roles)

        # a simple prompt adds the agent completions to the start of the prompt
        if completion_merge_type == "simpleprompt":
            # use the text from compeltions
            my_agent_completions = agent_completion_manager.get_completion_byId(agent_all_matched_ids)
            if len(my_agent_completions) > 0:  
                my_text = my_agent_completions[0].format_completion_text()
                content_text = my_text + content_text   
            matched_messages_agent = []
 



        matched_messages_account = []
        # does this agent use an open ai model to reduce and select only relevant prompts
        if use_prompt_reduction:
            # this will work on related prompts - not the latest prompts
            # later we need to determine if the request actually relates to a previous conversation
            account_matched_ids = self.get_matched_ids(account_completion_manager, context_type, "semantic", num_relevant_conversations, num_past_conversations)
            account_latest_ids = self.get_matched_ids(account_completion_manager, "latest", "semantic", num_relevant_conversations, num_past_conversations)
            
            #matched_messages_account_relevant = account_completion_manager.get_completion_messages(account_matched_ids, roles_used_in_context)
            matched_messages_account_latest = account_completion_manager.get_completion_messages(account_latest_ids, roles_used_in_context) 
            
            text_info = account_completion_manager.get_transcript(account_matched_ids, roles_used_in_context)
            logging.info(f'PromptBuilder text_info: {text_info}')

            preset_values = [text_info, content_text]
            my_useful_response = self.preset_handler.process_preset_prompt_values("getrelevantfacts", preset_values)
            useful_reponse = self.get_data_item(my_useful_response, "Useful information:")

            if useful_reponse != 'NONE' :
                matched_messages_account = matched_messages_account_latest +  [Message('system', useful_reponse)]
                logging.info(f'Useful information: {useful_reponse}')
        else:
            account_matched_ids = self.get_matched_ids(account_completion_manager, context_type, content_text, num_relevant_conversations, num_past_conversations)
            matched_messages_account = account_completion_manager.get_completion_messages(account_matched_ids, roles_used_in_context) 

        # get the relevant context if any
        context_text = ""
        if context_name != "" and context_name != "none":
            context_mgr =ContextManager(self.config)
            context = context_mgr.get_context(account_name, context_name)
            context_text = context.context_formated_text2("compact")

        agent_message_dicts = Message.get_list_of_dicts(matched_messages_agent)
        account_message_dicts = Message.get_list_of_dicts(matched_messages_account)

        my_user_content =  Message('user', content_text) 
        
        #Recall that for the completion API we need to put the user content at the end of the prompt
        if context_text == "":
            full_prompt = agent_message_dicts + account_message_dicts + my_user_content.as_list_of_dicts()
        else:
            my_user_context = Message('user', context_text)
            full_prompt = agent_message_dicts + my_user_context.as_list_of_dicts() + account_message_dicts + my_user_content.as_list_of_dicts()

        return full_prompt

    def get_matched_ids(self, completion_manager: CompletionManager, context_type : str, content_text : str, num_relevant_conversations : int, num_past_conversations : int):
        matched_accountIds = []

        if context_type == "keyword":
            matched_accountIds = completion_manager.find_keyword_promptIds(content_text, 'or', num_relevant_conversations)
        elif context_type == "keyword_match_all":
            matched_accountIds = completion_manager.find_keyword_promptIds(content_text,'and', num_relevant_conversations)
        elif context_type == "semantic":
            matched_accountIds = completion_manager.find_closest_completion_Ids(content_text, num_relevant_conversations, 0.1)
        elif context_type == "hybrid":
            matched_prompts_closest = completion_manager.find_closest_completion_Ids(content_text, num_relevant_conversations, 0.1)
            matched_prompts_latest = completion_manager.find_latest_completion_Ids(num_past_conversations)
            matched_accountIds = matched_prompts_closest + matched_prompts_latest
        elif context_type == "latest":
            matched_accountIds = completion_manager.find_latest_completion_Ids(num_past_conversations)
        else:
            matched_accountIds = []

        # sort the account prompts
        distinct_list = list(set(matched_accountIds))
        sorted_list = sorted(distinct_list, reverse=True)

        return sorted_list

    def get_data_item(self, input_string, data_point):
        regex_pattern = f"{re.escape(data_point)}\\s*(.*)"
        result = re.search(regex_pattern, input_string)
        if result:
            return result.group(1).strip()
        else:
            return None
     
