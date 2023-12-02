import os
import openai
from time import sleep

openai.api_key = os.getenv("OPENAI_API_KEY")

chat_models = ["gpt-4", "gpt-3.5-turbo"]
message_history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a unique, surprising, extremely randomized story with highly unpredictable changes of events."}
]

completion_models = ["text-davinci-003", "text-davinci-001", "davinci-instruct-beta", "davinci"]
prompt = "[System: You are a helpful assistant]\n\nUser: Write a unique, surprising, extremely randomized story with highly unpredictable changes of events.\n\nAI:"

results = []

# Testing chat models
for model in chat_models:
    sequences = set()
    for _ in range(30):
        completion = openai.ChatCompletion.create(
            model=model,
            messages=message_history,
            max_tokens=256,
            temperature=0
        )
        sequences.add(completion.choices[0].message['content'])
        sleep(1)
    print(f"\nModel {model} created {len(sequences)} unique sequences:")
    for seq in sequences:
        print(seq)
    results.append((len(sequences), model))

# Testing completion models
for model in completion_models:
    sequences = set()
    for _ in range(30):
        completion = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=256,
            temperature=0
        )
        sequences.add(completion.choices[0].text)
        sleep(1)
    print(f"\nModel {model} created {len(sequences)} unique sequences:")
    for seq in sequences:
        print(seq)
    results.append((len(sequences), model))

# Printing table of results
print("\nTable of Results:")
print("Num_Sequences\tModel_Name")
for num_sequences, model_name in results:
    print(f"{num_sequences}\t{model_name}")
