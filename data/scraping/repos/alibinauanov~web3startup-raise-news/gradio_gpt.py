import openai
import gradio
import api_key

openai.api_key = api_key.APIKEY

messages = [{"role": "system", "content": "You are a psychologist"}]

def CustomChatGPT(user_input):
    messages.append({"role":"user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    return reply

demo = gradio.Interface(fn=CustomChatGPT, inputs = 'text', outputs = 'text', title = "Your title")

demo.launch(share=True)