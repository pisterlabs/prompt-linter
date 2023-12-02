from .agent import BaseAgent, Task, CodeFile
from langchain.chat_models import ChatOpenAI


def run_write_code_chain(model: ChatOpenAI, task_definition: str) -> str:
    pass


class SoftwareEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def write_code(self, task: Task) -> CodeFile:
        pass

    def review_code(self, task: Task, file: CodeFile) -> CodeFile:
        pass


