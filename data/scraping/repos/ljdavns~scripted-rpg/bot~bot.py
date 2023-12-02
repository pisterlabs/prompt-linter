from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage
)
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
import os
import set_keys
import config
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import threading
import queue
from langchain.callbacks.base import BaseCallbackManager

model_token_mapping = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16000,
    "gpt-3.5-turbo-0301": 4096,
    "text-ada-001": 2049,
    "ada": 2049,
    "text-babbage-001": 2040,
    "babbage": 2049,
    "text-curie-001": 2049,
    "curie": 2049,
    "davinci": 2049,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
}

base_messages = [
    SystemMessage(content="You are now a rpg game host, your goal is to provide user(player) an interactive text adventure. \
        You should update the story or the game process based on the following detailed command and user(player)'s interaction. \
        You should output in Chinese. \
    ")
]
#clone a chat_history with base_messages
user_chat_history = {}

class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)

def bot_generate_stream(**kwargs):
    g = ThreadedGenerator()
    kwargs['streaming'] = True
    kwargs['callback_manager'] = BaseCallbackManager([ChainStreamHandler(g)])
    kwargs['thread_generator'] = g
    threading.Thread(target=bot_generate, kwargs=kwargs).start()
    return g

def bot_generate(game_id: str, message: str, history_enabled=True, write_to_history=True, temperature=0.7, message_type='human', chat_history=[], streaming=False, callback_manager=None, thread_generator:ThreadedGenerator=None):
    if len(chat_history) > 0:
        user_chat_history[game_id] = chat_history
    if message_type == 'human':
        new_message = HumanMessage(content=message)
    elif message_type == 'system':
        new_message = SystemMessage(content=message)
    elif message_type == 'AI':
        new_message = AIMessage(content=message)
    else:
        raise ValueError("message_type should be one of 'human', 'system' or 'AI'")
    chat = ChatOpenAI(model_name=os.environ['LOCAL_MODEL_NAME'] if 'LOCAL_MODEL_NAME' in os.environ else ("gpt-4" if config.GPT4_ENABLED else "gpt-3.5-turbo-16k"), temperature=temperature, streaming=streaming, callback_manager=callback_manager)
    try:
        if history_enabled:
            if game_id not in user_chat_history:
                user_chat_history[game_id] = base_messages.copy()
            chat_history = user_chat_history[game_id]
            AI_message = chat(chat_history + [new_message])
            if write_to_history:
                chat_history.append(new_message)
                chat_history.append(AI_message)
            # update_history(chat, game_id)
        else:
            AI_message = chat([new_message])
    finally:
        if streaming:
            thread_generator.close()
    return AI_message.content 

def get_chat_history(game_id='GAME1', limit=0):
    # convert the message objects to `xxMessage`:`content``
    if game_id not in user_chat_history:
        return []
    chat_history = user_chat_history[game_id]
    return chat_history[-limit:]

def get_chat_history_content(game_id='GAME1', limit=0):
    # convert the message objects to `xxMessage`:`content``
    if game_id not in user_chat_history:
        return []
    chat_history = user_chat_history[game_id]
    chat_history_content = ["{}:{}".format(type(message).__name__, message.content) for message in chat_history]
    return chat_history_content[-limit:]

def clear_chat_history(game_id='GAME1'):
    if game_id not in user_chat_history:
        return
    user_chat_history[game_id] = base_messages.copy()

def reset_chat_history(game_id='GAME1', chat_history=[]):
    user_chat_history[game_id] = chat_history

#todo
def update_history(model: ChatOpenAI, game_id):
    if game_id not in user_chat_history:
        return
    chat_history = user_chat_history[game_id]
    total_chat_history_length = model.get_num_tokens_from_messages(chat_history)
    if total_chat_history_length > model_token_mapping[model.model_name]:
        chat_history_as_text = [message.content for message in chat_history[1:]]
        user_chat_history[game_id] = base_messages.copy()
        return chat_history_as_text

if __name__ == "__main__":
    human_message = "你好，请为我生成一个rpg游戏的详细开头"
    AI_message = bot_generate('GAME1', human_message, streaming=True)
    print(get_chat_history())
    print("AI: " + AI_message)
