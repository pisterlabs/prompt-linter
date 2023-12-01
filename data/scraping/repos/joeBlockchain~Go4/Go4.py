import openai
import os
import datetime

API_KEY_FILE = 'api_key.txt'
GPT_MODEL_NAME = 'gpt-3.5-turbo'
SYSTEM_MESSAGE = "You are a helpful assistant."

conversation = []

def check_api_key(api_key):
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=GPT_MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_MESSAGE}]
        )
        return True
    except:
        return False

def save_api_key(api_key):
    with open(API_KEY_FILE, 'w') as f:
        f.write(api_key)

if not os.path.exists(API_KEY_FILE):
    api_key = input('API Key Needed. Please input your OpenAI API Key: ')
    while not check_api_key(api_key):
        api_key = input('Invalid API Key. The API key you entered is not valid. Please try again: ')
    save_api_key(api_key)
else:
    with open(API_KEY_FILE, 'r') as f:
        api_key = f.read().strip()
    if not check_api_key(api_key):
        api_key = input('Invalid API Key. The API key you entered is not valid. Please try again: ')
        while not check_api_key(api_key):
            api_key = input('Invalid API Key. The API key you entered is not valid. Please try again: ')
        save_api_key(api_key)

openai.api_key = api_key

def get_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%m/%d/%y %I:%M %p")

def make_api_call(messages):
    try:
        response = openai.ChatCompletion.create(
            model=GPT_MODEL_NAME,
            messages=messages
        )
        return response
    except Exception as e:
        print(f"Failed to get a response from the ChatGPT API: {str(e)}")
        return None

def send_message():
    user_input = input("\nYou: ")
    conversation.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + conversation
    response = make_api_call(messages)
    if response is not None and 'choices' in response and response["choices"][0]["message"]["content"]:
        assistant_reply = response["choices"][0]["message"]["content"]
        conversation.append({"role": "assistant", "content": assistant_reply})
        print(f"\n({get_timestamp()}) Go4: {assistant_reply}")

# Start the conversation
while True:
    send_message()
