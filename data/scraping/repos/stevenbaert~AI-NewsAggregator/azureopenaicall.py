import openai
import os
from dotenv import load_dotenv


def call_azureopenai(user_instruction=None, text=None):
    # Load environment variables
    load_dotenv()
    load_dotenv(override=True)

    # Initialize parameters
    params = {
        "engine": os.getenv("ENGINE", "gpt-4-32k"),
        "temperature": float(os.getenv("TEMPERATURE", 0.7)),
        "max_tokens": int(os.getenv("MAX_TOKENS", 800)),
        "frequency_penalty": float(os.getenv("FREQUENCY_PENALTY", 0)),
        "presence_penalty": float(os.getenv("PRESENCE_PENALTY", 0)),
        "stop": os.getenv("STOP", None),
    }

    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}

    # System context
    system_context = os.getenv(
        "SYSTEM_CONTEXT", "You are an AI assistant that helps people find information."
    )

    # Initialize messages list with system context
    messages = [{"role": "system", "content": system_context}]

    # Add text if provided
    if text:
        messages.append({"role": "user", "content": text})

    # Add user_instruction if provided
    if user_instruction:
        messages.append({"role": "user", "content": user_instruction})

    # OpenAI API settings
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")

    try:
        # Make API call
        response = openai.ChatCompletion.create(messages=messages, **params)
        return response
    except Exception as e:
        return f"An error occurred: {e}"


# Example usage
# Uncomment to run as standalone script
# if __name__ == "__main__":
#     print(chat_completion("Summarize this text.", "Piggy went to the market."))
