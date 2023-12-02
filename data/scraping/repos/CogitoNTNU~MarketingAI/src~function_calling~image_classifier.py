import logging
from langchain.llms.openai import OpenAI
from langchain.tools import StructuredTool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from src.gpt.text_generator import request_chat_completion
from src.config import Config

logger = logging.getLogger(__name__)


def get_image_template(user_prompt: str, classification: str) -> str:
    """
    Generate image template based on classification.
    Args:
        user_prompt: User prompt for image.
        classification: Classification of image. Recognized classifications are: meme, propaganda, marketing.
    """
    if classification == "propaganda":
        image_prompt = "Classic propaganda poster: Bold, primary colors" + user_prompt
    elif classification == "marketing":
        image_prompt = "Marketing material: Bright, primary colors. " + user_prompt
    elif classification == "meme":
        image_prompt = "Meme: " + user_prompt
    else: 
        image_prompt = "Poster: " + user_prompt
    return image_prompt

def classify_text(text: str) -> str:
    """Classify text into one of three categories: meme, propaganda, marketing."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    # Use gpt to classify 
    gpt_str = "Classify this text into one of three categories: meme, propaganda, marketing. \"" + text + "\". Response should be one of the three categories."
    result = request_chat_completion(previous_message={}, message=gpt_str)

    return "Classify this text into one of three categories: meme, propaganda, marketing. \"" + result + "\". Response should be one of the three categories."

tools: list[StructuredTool] = [
    StructuredTool.from_function(
        name= "Classify Text", 
        func=classify_text, 
        description="Classify text into one of three categories: meme, propaganda, marketing.",
    ),
]

# Make a memory for the agent to use
memory = ConversationBufferMemory(memory_key="chat_history")

llm = OpenAI(temperature=0, openai_api_key=Config().API_KEY)
agent_chain = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=False, 
    memory=memory,
    max_iterations=10,
    )

def run_agent(prompt: str) -> str:
    """Run the agent."""
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")

    if (len(prompt) < 1) or (len(prompt) > 1000):
        raise ValueError("Prompt must be at least 1 character or less than 1000 characters.")
    
    result = agent_chain.run(prompt)
    logger.info(f"Finished running langchain_function_calling.py, result: {result}")
    return result
