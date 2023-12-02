import openai
import gradio
import config_data
import logging 

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

openai.api_key = config_data.openapi_key

messages = [{"role": "system", 
             "content": "You are a helpful customer care assistant for an e-commerce website dealing with complaints from users. Please only answer complaint-related queries. Help the users with replacement or refund process. Reply with short messages"}]

def CustomChatGPT(user, name):
    messages.append({"role": "user", "content": user})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    logging.info(response, messages)

    ChatGPT_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return f"{name}: {ChatGPT_reply}"  # Include the chatbot's name in the output

def message_and_history(input, history, name):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    output = CustomChatGPT(inp, name)  # Pass the name to the CustomChatGPT function
    history.append((input, output))
    return history, history


block = gradio.Blocks(title="AI Chatbot", theme=gradio.themes.Monochrome())
with block:
    gradio.Markdown(
        """
    <div style="text-align:center;">
        <h1>Customer Service Assistant</h1>
        <p>Ask any any complaint relaed queries</p>
    </div>
    """
    )
    with gradio.Row():
        message = gradio.Textbox(label="Input Text")
        chatbot = gradio.Chatbot(default_response="Liza: Hello! I'm here to assist you with complaints.")
        #message = gradio.Textbox(placeholder=messages)
        #chatbot = gradio.Chatbot()
        state = gradio.State()
        message.submit(message_and_history,
                 inputs=[message, state],
                 outputs=[chatbot, state])
        message.submit(lambda: '', None, message)

        gradio.HTML("""
        <div style="display: flex; justify-content: center; margin-top: 20px;">
        <img src='/file=female.jpg' style='width: 150px; height: 190px;'>
        </div>
        <h2 style="text-align: center;">Liza</h2>  <!-- Display the chatbot's name -->
        """)

    
block.launch(debug=True, share=True)
