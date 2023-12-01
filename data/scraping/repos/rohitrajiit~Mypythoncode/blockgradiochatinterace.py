import gradio as gr
from openai import OpenAI

api_key = "sk-"  # Replace with your key

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        history_openai_format = []
        for human, assistant in chat_history:
            history_openai_format.append({"role": "user", "content": human })
            history_openai_format.append({"role": "assistant", "content":assistant})
        history_openai_format.append({"role": "user", "content": message})


        client = OpenAI(
        api_key=api_key,)
        response = client.chat.completions.create(
        messages=history_openai_format,
        # model="gpt-3.5-turbo" # gpt 3.5 turbo
        # model="gpt-4",
        model = "gpt-4-1106-preview", #gpt-4 turbo
        stream = False
        )
        responsetext = response.choices[0].message.content
        chat_history.append((message, responsetext))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
