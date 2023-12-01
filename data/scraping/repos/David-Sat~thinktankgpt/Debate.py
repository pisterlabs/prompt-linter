from langchain.schema import ChatMessage
from langchain.chat_models import ChatOpenAI
from typing import List, Dict, Optional, Any
from utils.Coordinator import Coordinator
from utils.Expert import Expert



class Debate():
    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo") -> None:
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.topic = None
        self.debate_history = []
        self.memory = []
        self.experts = []

    def add_message(self, role: str, content: str, avatar: Optional[str] = None) -> None:
        self.debate_history.append({"role": role, "avatar": avatar, "content": content})
        role = "user" if role == "user" else "assistant"
        self.memory.append(ChatMessage(role=role, content=content))


    def get_experts(self) -> List[Expert]:
        return self.experts

    def initialize_new_debate(self, topic: str, num_experts: int, stance: str) -> None:
        self.topic = topic
        expert_instructions = self.create_expert_instructions(num_experts, stance)
        self.experts = self.generate_experts(expert_instructions)

    def initialize_existing_debate(self, topic: str, debate_history: List[Dict[str, str]], expert_instructions: List[Dict[str, str]]) -> None:
        self.topic = topic
        self.debate_history = debate_history
        for message in debate_history:
            role = "user" if message["role"] == "user" else "assistant"
            self.memory.append(ChatMessage(role=role, content=message["content"]))
        self.experts = self.generate_experts(expert_instructions)

    def create_expert_instructions(self, num_experts: int, stance: str) -> List[Dict[str, str]]:
        coordinator_model = ChatOpenAI(openai_api_key=self.openai_api_key, model_name=self.model_name)
        coordinator = Coordinator(coordinator_model, num_experts, self.topic, stance)
        return coordinator.generate_expert_instructions()

    def generate_experts(self, experts_instructions: List[Dict[str, str]]) -> List[Expert]:
        experts = []
        for expert_instruction in experts_instructions:
            expert_model = ChatOpenAI(openai_api_key=self.openai_api_key, model_name=self.model_name, streaming=True)
            experts.append(Expert(expert_model, expert_instruction))
        return experts
    
    def get_debate_params(self) -> Dict[str, Any]:
        return {"topic": self.topic, "debate_history": self.debate_history, "expert_instructions": [expert.expert_instruction for expert in self.experts]}