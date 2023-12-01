import argparse

from friend_replica.format_chat import ChatConfig
from langchain.llms import GPT4All
from models.model_cn import ChatGLM
from friend_replica.recollection import LanguageModelwithRecollection
from friend_replica.semantic_search import *

parser = argparse.ArgumentParser()
parser.add_argument('--my_name', type=str, help='Your name in chat history.', default='Rosie')
parser.add_argument('--friend_name', type=str, help='Friend\'s name in chat history.', default='王')
parser.add_argument('--language', type=str, help='Choose between english and chinese.', default='chinese')
parser.add_argument('--device', type=str, help='Choose your device: cpu/cuda/mps...', default='cpu')
parser.add_argument('--debug', type=bool, help='Whether to print debugging information.', default=False)
args = parser.parse_args()

if args.language == 'chinese':
    model = ChatGLM()
else: 
    model = GPT4All(model="llama-2-7b-chat.ggmlv3.q4_0.bin")

chat_config = ChatConfig(
    my_name=args.my_name,
    friend_name=args.friend_name,
    language=args.language,
)
chat_with_friend = Chat(device=args.device, chat_config=chat_config)
m = LanguageModelwithRecollection(model, chat_with_friend)

q = ''
current_chat = []
while True:
    q = input("Chat with your friend now! To Exit, type \"exit\" or simply use ctrl C\n")
    if q == 'exit':
        break
    a = m(q, '\n'.join(current_chat))
    print(a)
    current_chat.append(chat_config.friend_name + ': ' + q)
    current_chat.append(a)