import os
import json
import openai
import time
from datetime import datetime

def load_role_from_file(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

def load_api_key_from_file(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

def chat_with_gpt3(prompt, role, conversation_history):
    openai.api_key = load_api_key_from_file("openaiapikey.txt")
    model = "gpt-3.5-turbo"
    messages = conversation_history + [{"role": role, "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        n=1,
        temperature=0.8,
    )

    return response.choices[0].message["content"]

def load_completed_tasks():
    if os.path.exists("completed_tasks.json"):
        with open("completed_tasks.json", "r") as file:
            return json.load(file)
    else:
        return {}

def update_completed_tasks(task):
    completed_tasks = load_completed_tasks()
    completed_tasks.update(task)

    with open("completed_tasks.json", "w") as file:
        json.dump(completed_tasks, file)

def main():
    os.makedirs("subjects", exist_ok=True)

    completed_tasks = load_completed_tasks()
    article_counter = 0

    for json_file in os.listdir("subjects"):
        if article_counter >= 2:
            break

        if json_file.endswith(".json") and json_file not in completed_tasks:
            with open(f"subjects/{json_file}", "r") as file:
                subjects = json.load(file)

            for subject in subjects:
                prompt_agent = load_role_from_file("promptagent.txt")
                prompt_monitor = load_role_from_file("promptmonitor.txt")

                conversation_history = [
                    {"role": "user", "content": "You are a helpful assistant."},
                ]

                for i in range(10):
                    if i % 2 == 0:
                        role = "user"
                        prompt = prompt_agent
                    else:
                        role = "assistant"
                        prompt = prompt_monitor

                    prompt = prompt.replace("<<SUBJECT>>", subject)

                    retries = 0
                    while retries < 3:
                        try:
                            response = chat_with_gpt3(prompt, role, conversation_history)
                            break
                        except openai.error.RateLimitError:
                            if retries < 2:
                                print("Rate limit error. Retrying in 10 seconds...")
                                time.sleep(10)
                                retries += 1
                            else:
                                raise

                    conversation_history.append({"role": role, "content": response})
                    print(f"{role}: {response}")

                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                output_filename = f"articles/technical_article_{timestamp}.txt"

                with open(output_filename, "w") as output_file:
                    for message in conversation_history:
                        output_file.write(f"{message['role']}: {message['content']}\n\n")

                print(f"Saved article to {output_filename}")

                update_completed_tasks({json_file: True})
                article_counter += 1
                if article_counter >= 2:
                    break

if __name__ == "__main__":
    main()
