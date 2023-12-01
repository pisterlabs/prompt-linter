import openai
import gradio as gr

openai.api_key = "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"

messages = [
    {"role": "system", "content": "A Genius AI that can respond to anything with ethics"},
]

def chatbot(input):
    if input:
        messages.append({"role": "user", "content": input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        return reply

inputs = gr.inputs.Textbox(lines=7, label="Prompt")
outputs = gr.outputs.Textbox(label="Response")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="mini gpt chabot",
             description="Ask anything !!",
             theme="compact").launch(share=True)
