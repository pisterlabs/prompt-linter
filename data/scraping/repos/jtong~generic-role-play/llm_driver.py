import openai


def build_step(**kwargs):
    def step(prompt):
        print("/====prompt===\\")
        print(prompt)
        print("\====prompt===/")

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )
        print("/====completion===\\")
        print(completion.choices)
        print("\====completion===/")
        return completion.choices[0].message.content
    return step


def build_chat(**kwargs):
    def chat(messages):

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            **kwargs
        )
        print("/====completion===\\")
        print(completion.choices)
        print("\====completion===/")
        return completion.choices[0].message.content
    return chat