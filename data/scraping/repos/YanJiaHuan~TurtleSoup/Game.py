import openai
import gradio as gr

openai.api_key = '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'
history = gr.outputs.State()
def chat_with_gpt3(prompt, history=gr.State('')):
    history.data = history.data + '\n' + 'User: ' + prompt
    response = openai.ChatCompletion.create(
      model="text-davinci-003.5",
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": history.data},
        ]
    )
    response_text = response.choices[0].message['content']
    history.data = history.data + '\n' + 'GPT-3: ' + response_text
    return response_text, history

iface = gr.Interface(fn=chat_with_gpt3,
                     inputs="text",
                     outputs=["text", gr.outputs.Textbox(label="History", type="auto", state=history)],
                     title="Chat with GPT-3",
                     description="You can chat with me just like you would with a human.")

iface.launch()

