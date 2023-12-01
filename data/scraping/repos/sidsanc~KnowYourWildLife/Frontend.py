import os
import gradio as gr
import numpy as np
import openai
from PIL import Image
from openai.api_resources import engine
from ImageProccesing import getImageTags
import config


openai.api_key = config.OPEN_AI_KEY

start_sequence = "\AI:"
restart_sequence = "\Human:"


def generate_response(prompt):
    completion = openai.Completion.create(
           model = "text-davinci-003",
           prompt = prompt,
           temperature = 0,
           max_tokens= 500, 
           top_p=1,
           frequency_penalty=0, 
           presence_penalty=0, 
           stop=[" Human:", " AI:"]
       ) 
    return completion.choices[0].text

def my_chatbot(input, history):
    history = history or []
    my_history = list(sum(history, ()))
    my_history.append(input)
    my_input = ' '.join(my_history)
    output = generate_response(my_input)
    history.append((input, output))
    return history, history 

def process_image(image):
    save_folder = "/desktop/uploads"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "uploaded_image.jpg")
    # Save the image array as a file
    image = image.astype(np.uint8)  # Convert to uint8
    pil_image = Image.fromarray(image)
    pil_image.save(save_path)

    return save_path

with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>Chatbot</center></h1>""")
    
    image_input = gr.inputs.Image(shape=(300, 200), label="Upload Image")
    path_output = gr.outputs.Textbox(label="Image Path")

    interface = gr.Interface(fn=process_image, inputs=image_input, outputs=path_output, title="Image Uploader")

    prompt = getImageTags()

    chatbot = gr.Chatbot()
    state = gr.State()
    txt = gr.Textbox(show_label=False, placeholder="Ask Something", value=prompt).style(container=False)
    txt.submit(my_chatbot, inputs=[txt, state], outputs=[chatbot, state])
    
demo.launch(share = True)