import openai
import gradio
import os

openai.api_base = "https://api.nova-oss.com/v1"
openai.api_key = "your API key"

messages = [{"role": "System", "content": "I Will help you in education"}] 

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

demo = gradio.Interface(fn=CustomChatGPT, inputs = "text", outputs = "text", title = "EDU AI")

demo.launch(share=True)
