import re

import openai as openai

import secret_stuff

openai.api_key = secret_stuff.API_KEY

def save(text):
    with open("requests.log", "a") as f:
        f.write(text + "\n")
        f.write("-----------------------------\n")

def get_embedding(text):
    response = openai.Embedding.create(
        input="Your text string goes here",
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    return embeddings

def get_importance(memory):
    prompt = "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory.\nMemory: " + memory + "\nRating:"
    save(prompt)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=2,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    try:
        return int(completion["choices"][0]["message"]["content"])
    except:
        try:
            return int(completion["choices"][0]["message"]["content"])[0]
        except:
            print("Oh no, can't evaluate importance well for memory " + memory)
            print("instead saying " + completion["choices"][0]["message"]["content"])
            return 5

def reflect_on_memories(name, memories, t):
    prompt = "Statements about " + str(name) + "\n"

    for idx, memory in enumerate(memories):
        prompt += str(idx) + ". " + memory.description + "\n"

    prompt += "What 5 high-level insights can you infer from the above statements? (example format: insight (because of {1, 6, 2, 3}))\n\n1."
    save(prompt)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=800,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    compl = completion["choices"][0]["message"]["content"]
    reflections = compl.split("\n")

    reflected = []

    for idx, reflection in enumerate(reflections):
        if idx == 0:
            actual_text = reflection
        else:
            actual_text = ". ".join(reflection.split(". ")[1:])
        print(actual_text)
        evidentiary = actual_text.split("(")[1].split(")")[0]
        evidence = tuple(int(x) for x in re.findall(r'\d+', evidentiary))
        reflected.append([actual_text.split("(")[0], evidence])

    return reflected

def summarize_core_memories(name, memories):
    prompt = "How would one describe " + name + "'s core characteristics given the following statements?\n"

    for memory in memories:
        prompt += "- " + memory.description + "\n"

    save(prompt)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=1200,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion["choices"][0]["message"]["content"]

def get_completion(prompt):
    save(prompt)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=1200,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # todo add some checks

    return completion["choices"][0]["message"]["content"]
