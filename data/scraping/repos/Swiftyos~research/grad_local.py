import gradio as gr
import openai


def predict(message, history):
    openai.api_base = "http://127.0.0.1:8081"
    openai.api_key = "asdas"

    history_transformer_format = history + [[message, ""]]

    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": item}
        for i, item in enumerate(history_transformer_format)
    ]

    response = openai.ChatCompletion.create(
        model="LocalLlama",
        messages=messages,
        temperature=0,
        stream=True,  # this time, we set stream=True
    )
    partial_message = ""
    for chunk in response:
        if 'content' in chunk['choices'][0]['delta']:
            partial_message += chunk['choices'][0]['delta']['content']
        yield partial_message

gr.ChatInterface(predict).queue().launch()
