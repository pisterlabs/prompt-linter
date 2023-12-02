import openai
import gradio
import config_data
import logging 

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

openai.api_key = config_data.openapi_key

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

messages = [
    {
        "role": "system",
        "content": "You are a helpful customer care assistant for an e-commerce website dealing with complaints from users. Please only answer complaint-related queries.",
    }
]

def limit_text(text, max_words):
    words = text.split()
    limited_words = words[:max_words]
    return " ".join(limited_words)

def CustomChatGPT(user):
    max_input_words = 30
    max_response_words = 30

    # Truncate user input if it exceeds the maximum word limit
    user_limited = limit_text(user, max_input_words)

    messages.append({"role": "user", "content": user_limited})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_response_words
    )
    logging.info(response, messages)

    ChatGPT_reply = response.choices[0].message.content
    ChatGPT_reply_limited = limit_text(ChatGPT_reply, max_response_words)
    messages.append({"role": "assistant", "content": ChatGPT_reply_limited})
    return ChatGPT_reply

def message_and_history(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = CustomChatGPT(inp)
    history.append((input, output))
    return history, history

block = gradio.Blocks(theme=gradio.themes.Monochrome())
with block:
    with gradio.Row():
        gradio.HTML(
            """
        <div style="display: flex; justify-content: center; margin-top: 20px;">
        <img src='/file=female.jpg' style='width: 150px; height: 190px;'>
        </div>
        """
        )
        message = gradio.Textbox(placeholder=messages)
        chatbot = gradio.Chatbot()
        state = gradio.State()
        message.submit(
            message_and_history, inputs=[message, state], outputs=[chatbot, state]
        )
        message.submit(lambda: "", None, message)

block.launch(debug=True, share=True)

