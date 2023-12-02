import sys
import os
from dotenv import load_dotenv
import json
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
    total_tokens = 0
    for message in conversation:
        tokens = tokenizer.encode(
            message['content'],
            disallowed_special=()
        )
        total_tokens += len(tokens)
    return total_tokens


class Agent:
    def __init__(self, name, prompt, history_file="history.json"):
        self.name = name
        self.history_file = history_file
        self.chat = ChatOpenAI(streaming=True,
                               callbacks=[StreamingStdOutCallbackHandler()],
                               temperature=1.0,
                               model="gpt-4")
        self.message_history = self.load_conversation()
        self.message_history.insert(0, SystemMessage(content=prompt))

    def message(self, user_message):
        with get_openai_callback() as cb:  # Use the context manager to count tokens
            self.message_history.append(HumanMessage(content=user_message))
            resp = self.chat(self.message_history)
            self.message_history.append(resp)
            self.save_conversation()

        # Access the total tokens used from the context manager
        #total_tokens = cb.total_tokens
        #print(f"Total tokens used for this message: {total_tokens}")

        return resp.content

    def save_conversation(self):
        with open(self.history_file, 'w') as f:
            # Convert Message objects to dictionaries and save as JSON
            history = [{'name': 'System', 'content': message.content} if isinstance(message, SystemMessage)
                    else {'name': 'User', 'content': message.content} if isinstance(message, HumanMessage)
                    else {'name': 'AI', 'content': message.content} 
                    for message in self.message_history]
            json.dump(history, f, indent=4)

    def load_conversation(self):
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

def append_to_json_file(filename, message, author, token_count):
    # Create a dictionary with the message data
    message_data = {
        "author": author,
        "datetime": datetime.now().isoformat(),
        "message": message,
        "tokens": token_count
    }
    
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
    open(filename, 'w').close()

def run_conversation(agent, json_filename="conversation.json"):
    clear_file(json_filename)  # Clear the conversation file
    conversation = []
    while True:
        user_message = input("You: ")
        agent_response = agent.message(user_message)
        print('\n')
        
        # Get token count for the user message
        user_token_count = tiktoken_len([{"content": user_message}])
        conversation.append({"name": "User", "content": user_message, "tokens": user_token_count})
        
        # Append the user message to the JSON file
        append_to_json_file(json_filename, user_message, "User", user_token_count)
        
        # Get token count for the agent response
        agent_token_count = tiktoken_len([{"content": agent_response}])
        conversation.append({"name": "Agent", "content": agent_response, "tokens": agent_token_count})
        
        # Append the agent response to the JSON file
        append_to_json_file(json_filename, agent_response, "Agent", agent_token_count)

if __name__ == "__main__":
    # Initialize the agent
    with open("System_prompt.txt") as f:
        sys_prompt = f.read()
    agent = Agent("Agent", sys_prompt)

    with open("Summarizer_sys_prompt.txt") as f:
        summarizer_sys_prompt = f.read()
    summarizer_agent = Agent("SummarizerAgent", summarizer_sys_prompt, history_file="summarizer_history.json")
    # Run the conversation
    run_conversation(agent)
