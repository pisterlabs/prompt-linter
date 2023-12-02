from reflexion.agents.programming import PythonReflexionAgent
from reflexion.datasets.programming import HumanEvalDataset
from reflexion.environments.programming import InternalTestingEnv, PythonTestingEnv
from reflexion.llms import OpenAIChatLLM

# Load a task from a dataset
LANG = "python"
dataset = HumanEvalDataset(language=LANG)
task_id, signature, docstring, tests = dataset[0]

# Instantiate an LLM
llm = OpenAIChatLLM(model_name="gpt-4", temperature=0)

# Instantiate a code execution environment
local_env = PythonTestingEnv(timeout=10)

# Instantiate a Reflexion agent with an internal testing environment
agent = PythonReflexionAgent(
    function_signature=signature,
    docstring=docstring,
    testing_env=InternalTestingEnv(
        function_signature=signature,
        docstring=docstring,
        language=LANG,
        local_env=local_env,
        llm=llm,
    ),
    llm=llm,
)

# Run the agent for a few steps
for _ in range(3):
    reward, message = agent.step()

# Evaluate the agent's implementation against the ground truth tests
rewards, messages = local_env.step(program=agent.implementation, tests=tests)
