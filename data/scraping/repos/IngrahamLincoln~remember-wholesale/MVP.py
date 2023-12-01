import sys
import os
from dotenv import load_dotenv
import pickle
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
    def __init__(self, name, prompt, history_file="history.pkl"):
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
        with open(self.history_file, 'wb') as f:
            pickle.dump(self.message_history, f)

    def load_conversation(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'rb') as f:
                return pickle.load(f)
        else:
            return []

def save_to_text_file(filename, conversation, token_count=None):
    with open(filename, 'w') as f:
        for message in conversation:
            f.write(f"{message['name']}: {message['content']}\n")
        if token_count is not None:
            f.write(f"Total tokens: {token_count}\n")

def run_conversation(agent, text_filename="conversation.txt"):
    conversation = []
    while True:
        user_message = input("You: ")
        agent_response = agent.message(user_message)
        print('\n')
        conversation.append({"name": "User", "content": user_message})
        conversation.append({"name": "Agent", "content": agent_response})
        
        # Calculate the total tokens for the conversation
        token_count = tiktoken_len(conversation)
        
        # Save the conversation with the latest token count appended
        save_to_text_file(text_filename, conversation, token_count)


if __name__ == "__main__":
    # Initialize the agent
    with open("System_prompt.txt") as f:
        sys_prompt = f.read()
    agent = Agent("Agent", sys_prompt)

    # Run the conversation
    run_conversation(agent)
