import openai
from autogen import oai

MODEL_NAME = "vicuna-7b-v1.5"

def create_text_completion():
    response = oai.Completion.create(
        config_list=[
            {
                "model": MODEL_NAME,
                "base_url": "http://localhost:8000/v1",
                "api_type": "open_ai",
                "api_key": "NULL",
            }
        ],
        prompt="Hi",
    )
    print(response)


def create_chat_completion():
    openai.api_base = "http://localhost:8000/v1"
    response = oai.ChatCompletion.create(
        config_list=[
            {
                "model": MODEL_NAME,
                "base_url": "http://localhost:8000/v1",
                "api_type": "open_ai",
                "api_key": "NULL",
            }
        ],
        messages=[{"role": "user", "content": "use C# write HELLO string"}]
    )
    print(response)

create_chat_completion()

