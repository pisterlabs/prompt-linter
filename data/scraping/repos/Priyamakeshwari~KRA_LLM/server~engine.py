# Import necessary libraries and modules
from langchain.llms import LlamaCpp, OpenAI
from langchain import PromptTemplate, LLMChain, ConversationChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import pandas as pd
import os

# Define the path to the pre-trained LlamaCpp model
MODEL_PATH = "./models/llama-2-13b-chat.Q5_K_M.gguf"

# Function to create a prompt template
def create_prompt() -> PromptTemplate:
    """
    Creates a prompt template for generating AI responses.

    Returns:
    - prompt (PromptTemplate): The prompt template used for AI response generation.
    """
    DEFAULT_PROMPT = """
     [IN <<SYS>>ST] You are a helpful, respectful, and best Axis Bank customer support assistant and your name is sara. Always do your best to assist customers with their inquiries. If you are unsure about an answer, truthfully say "I don't know" and remember the chat history {chat_history} <</SYS>> {human_input} [/INST]
    """
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=DEFAULT_PROMPT,
    )
    return prompt

# Function to load the LLMChain model
def load_model() -> LLMChain:
    """
    Loads the LLMChain model for AI-based interactions.

    Returns:
    - llm_chain (LLMChain): The loaded LLMChain model configured for AI interactions.
    """
    # Create a callback manager with a streaming callback
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Create a conversation memory
    memory = ConversationBufferWindowMemory(k=2, return_messages=True, memory_key="chat_history")
    
    # Initialize the LlamaCpp model
    llm_model = LlamaCpp(
        model_path=MODEL_PATH,
        max_tokens=4096,
        temperature=0.9,
        top_p=1,
        callback_manager=callback_manager,
        verbose=False,
        n_batch=512,
        n_gpu_layers=40,
        f16_kv=True,
    )
    
    # Create a prompt template
    prompt = create_prompt()

    # Initialize the LLMChain
    llm_chain = LLMChain(llm=llm_model, prompt=prompt, memory=memory, verbose=True)
    return llm_chain

# Function to query a database based on user input
def query_db(user_text):
    """
    Queries a database based on user input and returns a response.

    Parameters:
    - user_text (str): The user's input text to be used for the database query.

    Returns:
    - response (str): The response generated based on the user's input and the database query.
    """
    # Set the OpenAI API key (replace with your actual API key)
    OPENAI_API_KEY = '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'
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    # Read data from CSV files (replace with your actual data sources)
    employees = pd.read_csv('./db/Employees.csv')
    persona = pd.read_csv('./db/Persona.csv')
    customer_employees = pd.read_csv('./db/Customers_Employees.csv')
    customers = pd.read_csv('./db/Customers.csv')
    contact_history = pd.read_csv('./db/contacthistory.csv')
    product_holding = pd.read_csv('./db/Product_Holding.csv')
    kra = pd.read_csv('./db/RM_KRAs.csv')

    # Initialize the agent with OpenAI and dataframes
    agent = create_pandas_dataframe_agent(
        OpenAI(temperature=0, max_tokens=1000),
        [employees],
        verbose=True,
    )

    # Create a conversation chain and get the response
    response = agent.run(user_text)
    return response

# Main entry point of the script
if __name__ == '__main__':
    # Load the LLMChain model
    llm_chain = load_model()
    
    # Read input from standard input (stdin)
    import sys
    for input_model in sys.stdin:
        # Run the LLMChain with the input model
        llm_chain.run(input_model)
