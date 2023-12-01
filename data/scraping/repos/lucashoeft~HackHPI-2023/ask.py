import openai
import os

# get var from environment
openai.api_key = os.environ.get("VALENTIN_OPENAI_API_KEY")

def ask(messages):

    try:
        request = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        messages.append({"role": "assistant", "content": request['choices'][0]['message']['content']}) 
                
        return messages

    except Exception as err:
        return "Error: " + str(err)

def test_ask():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is 3 * 5?"
        }
    ]

    print(ask(messages))