import openai

import config

def get_completion(prompt, model="gpt-3.5-turbo", temperature = 0, messages = None):
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        # Temperature is the degree of randomness of the model's output
        # 0 would be same each time. 0.7 or 1 would be difference each time, and less likely words can be used:
        temperature=temperature,
    )
    return response.choices[0].message["content"]

def send_prompt(prompt, show_input = True, show_output = True, temperature = 0):
    if show_input:
        print("=== INPUT ===")
        print(prompt)

    response = get_completion(prompt, temperature=temperature)

    if show_output:
        print("=== RESPONSE ===")
        print(response)

    return response

def send_prompt_messages(messages, temperature = 0):
    last_message = messages[-1:]
    if config.is_debug:
        print("=== LAST MESSAGE ===")
        print(last_message)
    rsp = get_completion(prompt=None, temperature=temperature, messages=messages)
    if config.is_debug:
        print("=== RESPONSE ===")
        print(rsp)
    return rsp

def next_prompt(prompt):
    if config.is_debug:
        return send_prompt(prompt, temperature=config.TEMPERATURE)
    return send_prompt(prompt, False, False, temperature=config.TEMPERATURE)
