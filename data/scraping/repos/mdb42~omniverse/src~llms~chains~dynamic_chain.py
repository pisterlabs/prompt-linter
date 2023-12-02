from langchain import PromptTemplate
from langchain.chains.llm import LLMChain

from src.llms.prompts.protocol_templates import ASSISTANT_TEMPLATE
from src.llms.prompts.protocol_templates import SESSION_TEMPLATE
from src.llms.prompts.protocol_templates import STORYTELLER_TEMPLATE
from src.llms.prompts.protocol_templates import TUTOR_TEMPLATE

from src.logger_utils import create_logger
from src import constants


class DynamicChain(LLMChain):
    logger = create_logger(__name__, constants.SYSTEM_LOG_FILE)
    internal_chains: list = []
    internal_prompt: str = ''
    current_protocol: str = ''
    current_format: str = ''
    current_tone: str = ''
    messages: list = []
    template: str = ''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.internal_chains = []
        self.current_protocol = "Assistant"
        self.current_format = "Natural"
        self.current_tone = "Natural"

    def set_protocol(self, role):
        self.logger.info(f"Setting chain role to {role}")
        self.current_protocol = role
        if self.current_protocol== "Assistant":
            self.template = ASSISTANT_TEMPLATE
        elif self.current_protocol== "Session":
            self.template = SESSION_TEMPLATE
        elif self.current_protocol== "Storyteller":
            self.template = STORYTELLER_TEMPLATE
        elif self.current_protocol== "Tutor":
            self.template = TUTOR_TEMPLATE
        self.prompt = PromptTemplate(input_variables=["chat_lines", "summary", "sentiment_analysis", "input", "ai_name", "user_name"],
                                           template=self.template)








