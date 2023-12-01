import openai
import gradio as gr
from gradio.components import Radio

openai.api_key = "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"

messages = [{"role": "system", "content": "You are GPT-4, answer questions \
    if only they are related to crypto currency else return 'it is out of my scope'."}]

def generate_response(prompt):
    mode = "question-answer"
    if mode == "question-answer":
        result = openai.ChatCompletion.create(model="gpt-3.5-turbo-0301",
                                 messages=messages + [{"role": "user", "content": prompt}])
        return result['choices'][0]['message']['content']

    elif mode == "prompt-to-sql":
        openai.api_type = "azure"
        openai.api_base = "https://test -digdata.openai.azure.com/"
        openai.api_version = "2022-12-01"
        openai.api_key = '1c60ef61808b4590b3c6c5d5c86be3ed'
        response = openai.Completion.create(
            engine="code-davinci-002",
            prompt=f"###  mySQL tables, with their properties:\n#\n# Employee(id, name, department_id)\n# Department(id, name, address)\n# Salary_Payments(id, employee_id, amount, date)\n#\n###\
                     {prompt}\n\nSELECT",
            temperature=0,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=1,
            stop=["#",";"])
        openai.api_key = "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"  
        return response.choices[0].text

inputs = [
    gr.inputs.Textbox(lines=5, label="Question/Prompt", placeholder="Type a question or SQL prompt here..."),
]

outputs = gr.outputs.Textbox(label="Answer")

title = "GPT-4"
description = "GPT-4 is a question answering system that can answer questions related to crypto currency. \
    It can also generate SQL queries from a prompt."

examples = [
    ["What is the price of Bitcoin?", "Question-Answer"],
    ["What is the price of Ethereum?", "Question-Answer"],
    ["What is the price of Dogecoin?", "Question-Answer"],
    ["What is the price of Cardano?", "Question-Answer"],
]

gr.Interface(generate_response, inputs, outputs, title=title, description=description, examples=examples).launch()