import openai
import io
import time

token_use = 0

def wrapper(func, args):
    return func(*args)

def generate_prompt(job, parameters = None):
    file = open("prompts/" + job + ".txt",mode='r')
    template = file.read()
    file.close()
    if parameters is None:
        return template
    return wrapper(template.format, parameters)

def call_openai(prompt, max_tokens = 256, temp = 0.7):
    global token_use
    response = None
    while response is None:
        try:
            completion = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              max_tokens=max_tokens,
              temperature = temp,
              messages=[
                {"role": "system", "content": prompt}
              ]
            )
            response = completion["choices"][0].message.content;
            tokens = completion['usage']['total_tokens']
        except Exception as err:
            print(err)
            time.sleep(1)
    return response

def generate_image(prompt, size = "256x256"):
    global token_use
    url = None
    while url is None:
        try:
            url = openai.Image.create(prompt=prompt, size=size)['data'][0]['url']
        except Exception as e:
            if str(e) == "Your request was rejected as a result of our safety system. Your prompt may contain text that is not allowed by our safety system.":
                # Rephrase
                completion = openai.Completion.create(
                    model="text-davinci-003",
                    temperature=0.7,
                    max_tokens=245,
                    top_p=1,
                    frequency_penalty=0.1,
                    presence_penalty=0,
                    prompt="Rephrase: " + prompt + '\n')
                prompt = completion["choices"][0]["text"].strip()
                tokens = completion['usage']['total_tokens']
                token_use += tokens
            else:
                print("\Exception: " + str(e))
                time.sleep(1)
    return url
