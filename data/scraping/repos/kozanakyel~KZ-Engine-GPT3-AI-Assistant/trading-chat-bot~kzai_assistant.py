import os
from dotenv import load_dotenv
from service_chains import ServiceConversationChain
from service_chains import ServiceSelectionChain

load_dotenv()

openai_api_key = os.getenv('T_OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key


from typing import Dict, List, Any

from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI

from symbol_generation_service import SymboGenerationPromptService
from trading_advisor import TradingAdvisor
from redis_pdf_index_service import IndexRedisService

class KzaiAssistant(Chain, BaseModel):
    """Controller model for the GptVerse Assistant."""
    
    stage_id = "1"

    conversation_history: List[str] = []
    current_conversation_stage: str = "1"
    service_selection_chain: ServiceSelectionChain = Field(...)
    service_conversation_utterance_chain: ServiceConversationChain = Field(...)
    conversation_stage_dict: Dict = {
        "1": "Introduction: Begin the conversation with a polite greeting and a brief introduction about the company and its services.",
        "2": "Discover Preferences: Ask the client about their hobbies, interests or other personal information to provide a more personalized service.",
        "3": "Education Service Presentation: Provide more detailed information about the education services offered by the company.",
        "4": "AI Trading Service Presentation: Provide more detailed information about the AI trading services offered by the company.",
        "5": "Close: Ask if they want to proceed with the service. This could be starting a trial, setting up a meeting, or any other suitable next step.",
        "6": "Company Info: Provide general information about company like what is company and what are purposes and aimed etc.",
        "7": "Trading Advice Service Presentation: Provide and give detailed trading advice about to asked specific coin or asset"
    }

    agent_name: str = "AI Assistant"
    agent_role: str = "Service Representative"
    company_name: str = "GptVerse"
    company_business: str = "GptVerse is a company dedicated to the metaverse. We provide education and AI trading services."
    company_values: str = "Our vision is to adapt people to the metaverse with AI processes, education, and AI trading systems, thereby helping people act like they are in a metaverse platform."
    conversation_purpose: str = "Choosing the right service for the client and showing them the best option. If the service is selected \
        then provide more detailed information about service."
    conversation_type: str = "Chatting"


    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    
    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    def determine_conversation_stage(self):
        conversation_stage_id = self.service_selection_chain.run(
            conversation_history='"\n"'.join(self.conversation_history),
            current_conversation_stage=self.current_conversation_stage,
        )
        # # testing purposes....!!!!!!
        # if conversation_stage_id == "1":
        #     self.conversation_history = []

        self.current_conversation_stage = self.retrieve_conversation_stage(
            conversation_stage_id
        )
        self.stage_id = conversation_stage_id

        # print(f"Conversation Stage: {self.current_conversation_stage}")

    def human_step(self, human_input):
        # process human input
        human_input = human_input + "<END_OF_TURN>"
        self.conversation_history.append(human_input)
        self.conversation_history = self.conversation_history[-5:]

    def step(self):
        self._call(inputs={})

    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the GptVerseAssistant."""
        
        # print(f"stage: {self.stage_id}")
        if self.stage_id == "6":
            # print("you are the company dtailed phase!!!!")
            redis_service = IndexRedisService()
            response_f1 = redis_service.response_f1_query(self.conversation_history[-1])  
            # print(f'last questions : {self.conversation_history[-1]}')
            response_f1 = response_f1 + " <END_OF_TURN>"
            self.conversation_history.append(response_f1)
            # print(f"{self.agent_name}: ", response_f1.rstrip("<END_OF_TURN>"))

        if self.stage_id == "4":
            # print("you are the ai trading dtailed phase!!!!")
            redis_service = IndexRedisService()
            response_f1 = redis_service.response_f1_query(self.conversation_history[-1])  
            # print(f'last questions : {self.conversation_history[-1]}')
            response_f1 = response_f1 + " <END_OF_TURN>"
            self.conversation_history.append(response_f1)
            # print(f"{self.agent_name}: ", response_f1.rstrip("<END_OF_TURN>"))
            
        if self.stage_id == "7":
            # print(f'last conversation , {self.conversation_history[-1]}')
            symbol = SymboGenerationPromptService.get_symbol(self.conversation_history[-1])
            # print(symbol)
            tradv = TradingAdvisor.get_advice(symbol)
            tradv = f'For the {symbol}: ' + tradv + " <END_OF_TURN>"
            self.conversation_history.append(tradv)
            # print(f"{self.agent_name}: ", tradv.rstrip("<END_OF_TURN>"))
        
            # Generate agent's utterance
        
        ai_message = self.service_conversation_utterance_chain.run(
                agent_name=self.agent_name,
                agent_role=self.agent_role,
                company_name=self.company_name,
                company_values=self.company_values,
                conversation_purpose=self.conversation_purpose,
                conversation_history="\n".join(self.conversation_history),
                conversation_stage=self.current_conversation_stage,
                conversation_type=self.conversation_type,
        )

        # Add agent's response to conversation history
        self.conversation_history.append(ai_message)
        self.conversation_history = self.conversation_history[-5:]

        # print(f"{self.agent_name} - base: ", ai_message.rstrip("<END_OF_TURN>"))
        return {}
    
    def get_response(self, chat: str):
        self.human_step(chat)
        self.determine_conversation_stage()
        self.step()
        return self.conversation_history[-1].rstrip("<END_OF_TURN>")

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "GptVerseAssistant":
        """Initialize the GptVerseAssistant Controller."""
        service_selection_chain = ServiceSelectionChain.from_llm(llm, verbose=verbose)
        service_conversation_utterance_chain = ServiceConversationChain.from_llm(
            llm, verbose=verbose
        )

        return cls(
            service_selection_chain=service_selection_chain,
            service_conversation_utterance_chain=service_conversation_utterance_chain,
            verbose=verbose,
            **kwargs,
        )
