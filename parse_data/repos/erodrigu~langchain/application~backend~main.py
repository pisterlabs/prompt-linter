from loguru import logger
from config.config_loader import load_config
from models.language_model_factory import LanguageModelFactory
from prompts.prompt_executor import PromptExecutor
from prompts.sample_prompts import generate_prompts, generate_chat_prompts
from utils.logger_config import configure_logger

# from langchain.prompts import PromptTemplate


# Define user_input here
user_input = """
"How does photosynthesis work?",
"Can you explain the theory of relativity?",
"Tell me about the history of Ancient Rome."
"""


def main(user_input=None):
    try:
        config = load_config()
        model = LanguageModelFactory.create_model(
            config["model_type"],
            config["model_name"],
            config["cache"],
            config["temperature"],            
            config["api_key"],
        )

        configure_logger()
        
        if "chat" in model.model_type:
            prompts = generate_chat_prompts(user_input)
        else:
            prompts = generate_prompts(user_input)

        prompt_executor = PromptExecutor(model, prompts)
        for i, output in enumerate(prompt_executor.execute_prompts()):
            logger.info(f"Prompt {i + 1} Output:")
            logger.info(output)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main(user_input)
