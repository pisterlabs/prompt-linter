import logging
import os
import sys
from dotenv import load_dotenv
from langchain.llms.octoai_endpoint import OctoAIEndpoint
from langchain import PromptTemplate, LLMChain

import time
from termios import tcflush, TCIFLUSH

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the current working directory
os.chdir(current_dir)

# Set up logging configuration
logging.basicConfig(level=logging.CRITICAL)

# Load environment variables
load_dotenv()

# This code is used to interactively ask questions to the language model.
# It uses the OctoAI cloud language model to generate responses.


def handle_exit():
    """Print a goodbye message and exit the program."""
    print("\nGoodbye!\n")
    sys.exit(1)


def ask():
    """Interactively ask questions to the language model."""
    print("Loading...")

    # Load necessary values from environment
    endpoint_url = os.getenv("ENDPOINT_URL")

    # Set up the language model and predictor
    llm = OctoAIEndpoint(
        endpoint_url=endpoint_url,
        model_kwargs={
        "model": "llama-2-7b-chat",
        "messages": [
            {
                "role": "system",
                "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            }
        ],
        "stream": False,
        "max_tokens": 256
        }
    )

    # Define a prompt template
    template = "{question}"
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Set up the language model chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Clear the screen
    os.system("clear")

    print("Ready! Let's start the conversation")
    print("Press Ctrl+C to exit\n")

    # Provide an example prompt and response
    example_question = "Who is Leonardo Davinci?"
    print(
        "Example \n\nPrompt:",
        example_question,
        "\n\nResponse:",
        llm_chain.run(example_question),
    )

    try:
        tcflush(sys.stdin, TCIFLUSH)
        while True:
            # Collect user's prompt
            user_prompt = input("\nPrompt: ")
            if user_prompt.lower() == "exit":
                handle_exit()

            # Generate and print the response
            start_time = time.time()
            response = llm_chain.run(user_prompt)
            end_time = time.time()
            elapsed_time = end_time - start_time
            response = str(response).lstrip("\n")
            print(f"Response({round(elapsed_time,1)} sec): " + response)
    except KeyboardInterrupt:
        handle_exit()


if __name__ == "__main__":
    ask()
