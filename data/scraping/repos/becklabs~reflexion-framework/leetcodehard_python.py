from reflexion.agents.programming import PythonReflexionAgent
from reflexion.datasets.programming import LeetCodeHardDataset
from reflexion.environments.programming import (
    InternalTestingEnv,
    PythonTestingEnv,
    LeetCodeTestingEnv,
)
from reflexion.llms import OpenAIChatLLM
from leetcode_env.types import ProgrammingLanguage
import logging  
logging.basicConfig(level=logging.INFO)

import dotenv   
dotenv.load_dotenv()

# Load a task from a dataset
LANG = "python"
LEETCODE_LANG = ProgrammingLanguage.PYTHON3
DATASET_PATH = "../data/leetcode-hard-uncontaminated-python3.jsonl"
dataset = LeetCodeHardDataset(DATASET_PATH)
task_id, signature, docstring, tests = dataset[0]

# Instantiate an LLM
llm = OpenAIChatLLM(model_name="gpt-4", temperature=0)

# Instantiate a code execution environment
local_env = PythonTestingEnv(timeout=10)

# Instantiate a Leetcode evaluation environment
leetcode_env = LeetCodeTestingEnv(language=LEETCODE_LANG, timeout=10)

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
for _ in range(1):
    reward, message = agent.step()

# Evaluate the agent's implementation against the ground truth tests
rewards, messages = leetcode_env.step(
    program=agent.implementation, metadata={"question_slug": task_id}
)
