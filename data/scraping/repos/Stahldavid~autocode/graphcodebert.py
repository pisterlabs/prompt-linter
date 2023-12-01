# Importar módulos do langchain
from langchain.agents import Tool, SharedMemory, AgentExecutor
from langchain.memory import Memory

# Importar o modelo GraphCodeBERT e o tokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Importar o agente CodeAgent
from code_agent import CodeAgent

tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModelForMaskedLM.from_pretrained("microsoft/graphcodebert-base")

class GraphCodeBertTool(Tool):
    def __init__(self, shared_memory: SharedMemory):
        super().__init__(actions={})
        self.shared_memory = shared_memory
        self.graphcodebert_model = model

    def predict_action(self, input: str) -> str:
        tool_name, query = input.split(":", 1)
        if tool_name == "#graphcodebert_tool":
            return query.strip()
        else:
            return None

    def execute_action(self, action: str) -> str:
        if action is not None:
            output = self.graphcodebert_model(action)
            return str(output)
        else:
            return None

# Exemplo de uso:

# Criar uma instância de SharedMemory para código
class CodeMemory(Memory):
    pass

shared_memory_code = SharedMemory(CodeMemory(), "code")

# Criar uma instância do GraphCodeBertTool com shared_memory_code
graphcodebert_tool = GraphCodeBertTool(shared_memory_code)

# Criar uma instância do AgentExecutor com graphcodebert_tool como parâmetro
agent_executor = AgentExecutor(
    agent=CodeAgent(prompt_template, language_model, stop_sequence, output_parser),
    memory=ConversationMemory(),
    tools=[graphcodebert_tool],
    max_turns=10
)

# Execute o AgentExecutor
agent_executor.run()
