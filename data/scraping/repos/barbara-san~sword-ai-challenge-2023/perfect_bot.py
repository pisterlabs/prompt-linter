from langchain.chat_models import AzureChatOpenAI

from config import load_environment
load_environment()

# You must also ask for the answer to be in a specific format, that best fits the question or request, for example, using bullet points or an ordered list. You MUST choose the formatation on your own. \


better_prompt_llm = AzureChatOpenAI(deployment_name="gpt35-team-3-0301", temperature= 0.2)

def perfect_prompt(prompt):
    return better_prompt_llm.predict(
        f"""\
        The AI is going to receive a prompt from the user, delimited by ```. \
        Please provide a revised prompt that adheres to the following guidelines: 
        1. The prompt should be understandable and comprehensive for other Large Language Models (LLMs) like OpenAI's GPT3.5. \
        2. The LLM should be aware that its task is to rephrase the given prompt in order to create a prompt that is easier to comprehend and more comprehensive for other LLMs, while maintaining the original prompt's objective. \
        3. The LLM must ensure that the revised prompt does not contain any questions, as its purpose is solely to generate a more understandable prompt for other LLMs. \
        Please generate a prompt that aligns with these specifications for the LLM to follow.
        The new prompt may include new additional text that is predictable to be presented by the human in the future. \
        \
        Prompt: \
        ```\
        {prompt}\
        ```\
        \
        You MUST only return the new prompt, and nothing else.\
        """
    )