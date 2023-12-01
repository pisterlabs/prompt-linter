# Import Azure OpenAI
import os
from langchain.llms import AzureOpenAI

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "..."
os.environ["OPENAI_API_KEY"] = "..."

# Create an instance of Azure OpenAI
# Replace the deployment name with your own
llm = AzureOpenAI(
    deployment_name="td2",
    model_name="text-davinci-002",
)

# Run the LLM
llm("Você conhece a Porto Seguro? Me fale sobre essa empresa.")