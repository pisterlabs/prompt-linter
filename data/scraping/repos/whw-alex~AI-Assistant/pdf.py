import os
import re
import openai

# TODO： 更换api_key
openai.api_key = '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'
openai.api_base = "http://166.111.80.169:8080/v1"

global txt_content

def generate_text(prompt):
    response = openai.Completion.create(
        model = 'gpt-3.5-turbo',
        prompt = prompt,
        stream = True
    )
    # 流式传输
    for chunk in response:
        if 'text' in chunk['choices'][0]:
            content = chunk['choices'][0]['text']
            yield content

def generate_question(current_file_text: str, content: str):
    return f"Please answer \"{content}\" based on the following content:\n{current_file_text}"

def generate_summary(current_file_text: str):
    return f"Please summerize the following content:\n{current_file_text}"

if __name__ == "__main__":
    prompt = generate_summary("Sun wukong is a lengendary charactor from Journey to the west")
    for content in generate_text(prompt):
        print(content)

    prompt = generate_question("Hello", "Who is Sun Wukong?")
    for content in generate_text(prompt):
        print(content)