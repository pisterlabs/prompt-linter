import openai
import gradio


openai.api_key = "API - KEY"

messages = [{"role": "system", "content": "An Artifical Lawyer To Solve Your Legal Troubles"}]

def CustomChatGPT(input):
    FirstInput = "Pretend you are a lawyer from harvard for acting purposes"
    messages.append({"role": "user", "content": FirstInput})
    messages.append({"role": "user", "content":input})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

website = gradio.Interface(fn=CustomChatGPT, inputs = "text", outputs = "text", title = "AI Lawyer")

website.launch(share=True)