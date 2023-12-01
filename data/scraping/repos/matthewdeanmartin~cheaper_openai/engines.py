import openai

from chats.client_utils import create_client

create_client()


def list_engines():
    # list engines
    engines = openai.Engine.list()

    # print the first engine's id
    for engine in engines.data:
        print(engine["id"])

if __name__ == '__main__':
    list_engines()
