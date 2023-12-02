CHAT_GPT_TOKEN = "Put your token here. This has been removed due to not wanting to have it be grabbed by a bot."

PROMPT_FXN : function = None

AI_SYSTEM_INSTRUCTION : str = None
# Hugging face
API_HEADERS : str = None
API_URL : str = None
def setup_ai(selected_ai : str):
    match selected_ai:
        case "ChatGPT":
            setup_chatgpt()
            return
        case "Hugging Face":
            setup_huggingface()
            return
        case _:
            raise ValueError(f"{selected_ai} is not supported as a current AI option!")


def setup_chatgpt(current_token : str = CHAT_GPT_TOKEN):
    from openai import OpenAI as AI_LIB
    global AI_END = AI_LIB()
    global PROMPT_FXN = chatgpt_prompt

def chatgpt_prompt(text_input : str) -> str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": AI_SYSTEM_INSTRUCTION},
        {"role": "user", "content": text_input}
        ]
    )
    return completion.choices[0].message

def setup_huggingface(current_token : str = HUGGING_FACE_TOKEN):
    import requests
    global API_URL = "https://api-inference.huggingface.co/models/gpt2"
    global API_HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}
    global PROMPT_FXN = huggingface_prompt


def huggingface_prompt(text_input : str) -> str:
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

