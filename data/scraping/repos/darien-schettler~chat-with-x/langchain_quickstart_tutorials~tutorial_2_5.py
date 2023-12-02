import argparse
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from misc.utils import retrieve_local_api_keys, print_colored_output


def chat_completion_script(model_name="gpt-3.5-turbo", temperature=0,
                           system_message="I am a chatbot. Ask me a question.",
                           input_clr="blue", response_clr="green", system_clr="magenta",
                           use_color=True, **kwargs):
    """ Generate a chat completion using LangChain ChatOpenAI

    Args:
        model_name (str, optional)): Model name for the Chatbot (default: gpt-3.5-turbo)
        temperature (float, optional)): Temperature for randomness in output. A value of 0 means no randomness.
        system_message (str, optional)): Message to display when the agent is thinking
        input_clr (str, optional)): Color for user input text in the chat
        response_clr (str, optional)): Color for AI response text in the chat
        system_clr (str, optional)): Color for system message text in the chat
        use_color (bool, optional)): Color only applied to the titles
        **kwargs: Additional keyword arguments

    Returns:
        None; prints the response to the query using the specified agent
    """

    # Load the API keys from the .env file
    retrieve_local_api_keys()

    # Initialize the ChatOpenAI instance with the specified parameters
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    # Set up prompt templates and conversation chain
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Initialize the conversation chain and memory buffer
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    # Continue the conversation loop until the user decides to exit
    while True:
        input_text = input("\nEnter your message (or press enter to exit):\n")
        if not input_text: break

        # Run the conversation with the given input
        response = conversation.predict(input=input_text)

        # Print the response
        print_colored_output(input_text, response, system_message,
                             input_color=input_clr,
                             response_color=response_clr,
                             system_color=system_clr,
                             full_color=use_color)


# Example Usage:
# python3 -m langchain_quickstart_tutorials/tutorial_2_5
#
# Enter your message (or press enter to exit):
# >>> [Ask an interesting question... after hearing the answer ask a followup that will demonstrate the memory ability]
#
# Alternative system message:
# 'You are an expert in Python programming and the best practices used to code. Please
# help the user generate any code that he/she requires. If you can't or don't know how, you should truthfully say you
# don't know. You should always include code to demonstrate your point or answer the query.'
def main():
    parser = argparse.ArgumentParser(description="Generate a conversation with memory using LangChain ChatOpenAI agent")
    parser.add_argument("-n", "--model_name", type=str, default="gpt-3.5-turbo", help="Model name for the ChatLLM")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Temperature for randomness in output")
    parser.add_argument("--system_message", type=str,
                        default="The following is a friendly conversation between a human and an AI. The AI is "
                                "talkative and provides lots of specific details from its context. If the AI does not "
                                "know the answer to a question, it truthfully says it does not know.",
                        help="System message to be used in the chat completion")
    parser.add_argument("-c", "--use_color", action="store_false", help="Color only to titles")
    parser.add_argument("--input_clr", type=str, default="blue", help="Color for user input text")
    parser.add_argument("--response_clr", type=str, default="green", help="Color for AI response text")
    parser.add_argument("--system_clr", type=str, default="magenta", help="Color for system message text")
    args = parser.parse_args()

    # Generate the chat completion based on the user inputs
    chat_completion_script(**args.__dict__)


if __name__ == "__main__":
    main()
