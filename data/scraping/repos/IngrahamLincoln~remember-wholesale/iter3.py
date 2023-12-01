import sys
import os
from dotenv import load_dotenv
import json
import uuid
from datetime import datetime
import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load environment variables
load_dotenv()
api_key = os.getenv('OPEN_API_KEY')

tokenizer = tiktoken.get_encoding('cl100k_base')

#length function for tokenizer
def tiktoken_len(conversation):
    """Calculates the total number of tokens in a conversation.

    Args:
        conversation (list of dict): A list of message dictionaries to calculate tokens for.

    Returns:
        int: The total token count for the conversation.
    """
    total_tokens = 0
    for message in conversation:
        tokens = tokenizer.encode(
            message['content'],
            disallowed_special=()
        )
        total_tokens += len(tokens)
    return total_tokens


class Agent:
    """Represents an agent that handles messages and maintains a conversation history.

    Attributes:
        name (str): The name of the agent.
        history_file (str): Path to the file where the conversation history is saved.
        chat (ChatOpenAI): An instance of the ChatOpenAI class used for generating responses.
        message_history (list): A list of messages that make up the conversation history.
    """

    def __init__(self, name, prompt, history_file="history.json"):
        """Initializes the Agent with a name, a starting prompt, and a history file.

        Args:
            name (str): The name of the agent.
            prompt (str): An initial system message or prompt to start the conversation.
            history_file (str): The file path for storing the conversation history.
        """     
        self.name = name
        self.history_file = history_file
        self.chat = ChatOpenAI(streaming=True,
                               callbacks=[StreamingStdOutCallbackHandler()],
                               temperature=1.0,
                               model="gpt-4")
        self.message_history = self.load_conversation()
        self.message_history.insert(0, SystemMessage(content=prompt))

    def message(self, user_message):
        """Processes a user message and generates a response.

        Args:
            user_message (str): The message input from the user.

        Returns:
            str: The generated response from the agent.
        """
        with get_openai_callback() as cb:  # Use the context manager to count tokens
            self.message_history.append(HumanMessage(content=user_message))
            resp = self.chat(self.message_history)
            self.message_history.append(resp)
            self.save_conversation()
        return resp.content
    def summarize(self, chunk):
        self.message_history.append(HumanMessage(content=chunk))
        resp = self.chat(self.message_history)
        return resp.content
        

    def save_conversation(self):
        """
        Saves the current state of the conversation to a JSON file.

        This method iterates through the message history, which consists of various message objects,
        and transforms them into a dictionary format suitable for JSON serialization. Each message
        is categorized by its type (System, User, AI) and contains relevant data such as content and token count.
        The resulting JSON structure is then written to the history file.
        """
        with open(self.history_file, 'w') as f:
            # Convert Message objects to dictionaries and save as JSON
            history = []
            for message in self.message_history:
                # Determine the type of message and construct the dictionary
                if isinstance(message, SystemMessage):
                    message_dict = {'name': 'System', 'content': message.content}
                elif isinstance(message, HumanMessage):
                    message_dict = {'name': 'User', 'content': message.content}
                else:  # Assuming AIMessage for simplicity
                    message_dict = {'name': 'AI', 'content': message.content}
                
                # Add the token count to the message dictionary
                message_dict['tokens'] = tiktoken_len([message_dict])
                
                # Append the message dictionary to the history list
                history.append(message_dict)
                
            json.dump(history, f, indent=4)

    def load_conversation(self):
        """
        Loads the conversation history from a JSON file.

        This method checks if the history file exists and is not empty, then reads the JSON content.
        It converts the JSON array back into a list of message objects, preserving the original types
        (System, User, AI). If the JSON is invalid or the file is empty, it returns an empty list
        and prints an error message.

        Returns:
            list: A list of message objects reconstructed from the JSON history file.
        """
        if os.path.exists(self.history_file) and os.path.getsize(self.history_file) > 0:
            with open(self.history_file, 'r') as f:
                try:
                    history = json.load(f)
                    return [SystemMessage(content=message['content']) if message['name'] == 'System' 
                            else HumanMessage(content=message['content']) if message['name'] == 'User' 
                            else AIMessage(content=message['content']) 
                            for message in history]
                except json.JSONDecodeError:
                    # Handle the empty file or invalid JSON structure
                    print(f"Error reading {self.history_file}. File is empty or not valid JSON.")
                    return []
        else:
            return []

def append_to_json_file(filename, message_data):
    """Appends a new message to a JSON file.

    If the file does not exist or is empty, it creates a new file and adds the message data.

    Args:
        filename (str): The name of the file to append the message data to.
        message_data (dict): The message data to append.
    """
    # Check if the file exists and contains data
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'r+') as f:
            # Load the existing data
            data = json.load(f)
            # Append the new message
            data.append(message_data)
            # Move the pointer (cursor) to the beginning of the file
            f.seek(0)
            # Convert back to JSON and write in the file
            json.dump(data, f, indent=4)
    else:
        # If the file doesn't exist or is empty, create a new list with the message
        with open(filename, 'w') as f:
            json.dump([message_data], f, indent=4)


def clear_file(filename):
    """Clears the content of the specified file.

    Args:
        filename (str): The name of the file to clear.
    """
    open(filename, 'w').close()



def check_to_summarize(filename="conversation.json", token_bound = 20):
    """Processes the messages in the history file and prepares them for summarization.

    Args:
        filename (str): The name of the file to process messages from.

    Returns:
        str: Concatenated string of all messages that have not been summarized.
    """
    # ... [omitted docstring and comments for brevity]

    # Load the message history
    with open(filename, 'r') as f:
        messages = json.load(f)

    # Calculate the total token count for unsummarized messages
    total_tokens = sum(msg['tokens'] for msg in messages if not msg.get('summarized', False))

    if total_tokens < token_bound:
        return False
    
    # Initialize an empty string to store all content if the token threshold is met
    all_content = ''

    # Only proceed if the total token count is 2000 or more
    if total_tokens >= token_bound:
        # Filter out messages that have been summarized or have empty content
        filtered_messages = [msg['content'] for msg in messages if not msg.get('summarized', False) and msg['content'].strip()]

        # Concatenate all the content fields into a single string
        all_content = ' '.join(filtered_messages)

        # Update the 'summarized' status for all messages that have been concatenated
        for msg in messages:
            if not msg.get('summarized', False) and msg['content'].strip():
                msg['summarized'] = True

    # Save the updated messages back to the file regardless of token count
    with open(filename, 'w') as f:
        json.dump(messages, f, indent=4)

    return all_content


def save_summary(summary):
    """
    Saves the summary with a unique identifier and timestamp.

    Args:
        summary (str): The summary text to be saved.

    Returns:
        dict: A dictionary with the summary id, timestamp, author, and the summary text.
    """
    # Each summary is stored with a timestamp
    summary_data = {
        "id": str(uuid.uuid4()),
        "datetime": datetime.now().isoformat(),
        "name": "summarizer agent",
        "content": summary,
    }
    return summary_data

def save_message(message, author, token_count):
    """Creates a dictionary to save message data.

    Args:
        message (str): The content of the message.
        author (str): The name of the author of the message.
        token_count (int): The token count of the message.

    Returns:
        dict: A dictionary containing the message data.
    """
    # Get random unique id
    message_id = str(uuid.uuid4())
    # Create a dictionary with the message data
    message_data = {
        "id": message_id,
        "name": author,
        "datetime": datetime.now().isoformat(),
        "content": message,
        "tokens": token_count,
        "summarized": False 
    }
    return message_data

def run_conversation(agent, summarizer_agent, convo_filename="conversation.json", summary_file="summarizer_history.json"):
    """
    The main loop for running the conversation, handling user input, generating agent responses, summarizing the conversation, and saving the messages and summaries.

    Args:
        agent (Agent): The conversational agent to interact with the user.
        summarizer_agent (Agent): The agent responsible for summarizing the conversation.
        convo_filename (str): The filename for saving the conversation.
        summary_file (str): The filename for saving summaries of the conversation.
    """   
    clear_file(convo_filename)  # Clear the conversation file
    conversation = []
    while True:
        user_message = input("You: ")
        agent_response = agent.message(user_message)
        print('\n')
        
        # Get token count for the user message
        user_token_count = tiktoken_len([{"content": user_message}])
        conversation.append({"name": "User", "content": user_message, "tokens": user_token_count})
        
        # Append the user message to the JSON file
        user_message_data = save_message(user_message, "User", user_token_count)
        append_to_json_file(convo_filename, user_message_data)
        
        # Get token count for the agent response
        agent_token_count = tiktoken_len([{"content": agent_response}])
        conversation.append({"name": "Agent", "content": agent_response, "tokens": agent_token_count})
        
        # Append the agent response to the JSON file
        agent_message_data = save_message(agent_response, "Agent", agent_token_count)
        append_to_json_file(convo_filename, agent_message_data)

        chunk = check_to_summarize()
        if isinstance(chunk, str) and chunk:
            summary = summarizer_agent.summarize(chunk)
            summary_data = save_summary(summary)
            append_to_json_file(summary_file, summary_data)



if __name__ == "__main__":
    # Initialize the agent
    with open("System_prompt.txt") as f:
        sys_prompt = f.read()
    agent = Agent("Agent", sys_prompt)

    with open("Summarizer_sys_prompt.txt") as f:
        summarizer_sys_prompt = f.read()
    summarizer_agent = Agent("SummarizerAgent", summarizer_sys_prompt, history_file="summarizer_history.json")
    # Run the conversation
    run_conversation(agent, summarizer_agent)
