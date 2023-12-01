import openai
import re
import dotenv
import os

dotenv.load_dotenv()

openai.api_key = os.environ.get("API_KEY")

def shopping_assistant(model="gpt-3.5-turbo", max_tokens=100):
    done = False
    response = input("What do you want to eat?\n")
    messages_history = [
        {"role": "system", "content": "You are a helpful assistant who will suggest a restaurant. Your goal is to determine types of food that people want. When you are done with your task, say DONE."},
        {"role": "assistant", "content": "What do you want to buy?"}]
    while not done:
        messages_history += [{"role": "user", "content": response}]

        output = openai.ChatCompletion.create(
            model=model,
            messages=messages_history,
            max_tokens=max_tokens
            )
        output_text = output['choices'][0]['message']['content']
        messages_history += [{"role": "assistant", "content": output_text}]

        response = input(output_text + "\n")

        done = True # somehow figure out when the tags have been given.

    tags = extract_tags(messages_history, model=model, max_tokens=max_tokens)

    return tags

def extract_tags(messages_history, model="gpt-3.5-turbo", max_tokens=100):
    messages_history = messages_history[1:] + [{"role": "system", "content": "Now you are a tag extraction system. Extract the given tags into the format of a python list of strings."}]
    output = openai.ChatCompletion.create(
            model=model,
            messages=messages_history,
            max_tokens=max_tokens
            )
    output_text = output['choices'][0]['message']['content']

    print(output_text)

    pattern = r"\[(.*?)\]"
    tags = re.findall(pattern, output_text)
    return tags

# tags = shopping_assistant()
# print(tags)