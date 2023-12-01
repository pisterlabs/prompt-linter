import openai
from secret import api_key

def test_connection() -> None:
    openai.api_key = api_key
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "system", "content": "This is a test, if you receive my message, just response OK."}], 
        temperature=0.7
    )
    print(chat.choices[0].message.content)

def send_message(message: str, temperature: float) -> str:
    openai.api_key = api_key
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "system", "content": message}], 
        temperature=temperature
    )
    return chat.choices[0].message.content

def send_message_with_role(prompt: str, message: str, temperature: float) -> str:
    openai.api_key = api_key
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": message}], 
        temperature=temperature
    )
    return chat.choices[0].message.content

def send_message_with_role_concatenate(prompt_system: str, prompt_begin: str, message: str, prompt_end: str, temperature: float) -> str:
    openai.api_key = api_key
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "system", "content": prompt_system}, 
                  {"role": "user", "content": prompt_begin + message + prompt_end}], 
        temperature=temperature
    )
    return chat.choices[0].message.content