from langchain.agents import Tool, ReadOnlySharedMemory, AgentExecutor
from langchain.memory import Memory
import torch
from code_agent import CodeAgent
from transformers import AutoTokenizer, AutoModel

class UnixcoderTool(Tool):
    def __init__(self, shared_memory: ReadOnlySharedMemory):
        super().__init__(actions={})
        self.shared_memory = shared_memory
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        self.unixcoder_model = AutoModel.from_pretrained("microsoft/unixcoder-base")
        self.unixcoder_model.to(device)

    def predict_action(self, input: str) -> str:
        tool_name, query = input.split(":", 1)
        if tool_name == "#unixcoder_tool":
            return query.strip()
        else:
            return None

    def execute_action(self, action: str) -> str:
        if action is not None:
            # Encode the input action using the tokenizer
            inputs = self.tokenizer(action, return_tensors="pt", padding=True)

            # Move the inputs to the device
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

            # Utilize o modelo UniXcoder baseado na tarefa especificada na ação
            with torch.no_grad():
                outputs = self.unixcoder_model(**inputs)

            return str(outputs)
        else:
            return None

# Exemplo de uso:

# Criar uma instância de SharedMemory para código
class CodeMemory(Memory):
    pass

shared_memory_code = ReadOnlySharedMemory(CodeMemory(), "code")

# Criar uma instância do UnixcoderTool com shared_memory_code
unixcoder_tool = UnixcoderTool(shared_memory_code)

# Criar uma instância do AgentExecutor com unixcoder_tool como parâmetro
agent_executor = AgentExecutor(
    agent=CodeAgent(prompt_template, language_model, stop_sequence, output_parser),
    memory=ConversationMemory(),
    tools=[unixcoder_tool],
    max_turns=10
)

# Execute o AgentExecutor
agent_executor.run()
