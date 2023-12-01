import streamlit as st
from openai import OpenAI
import consts
import os
import tempfile
import io
import base64
from PIL import Image
import math

os.environ['OPENAI_API_KEY'] = consts.API_KEY_OPEN_AI
client = OpenAI()
st.title("AI-powered Picture Discussion")  

def get_image_base64(image_bytes):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')  # You can change to 'JPEG' if needed
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

def calculate_new_dimensions(width, height, max_dimension=512):
    ratio = width / height
    if width > height:
        temp_width = max_dimension
        temp_height = temp_width / ratio
    else:
        temp_height = max_dimension
        temp_width = temp_height * ratio
    # Math.ceil() rounds up to the nearest integer
    return math.ceil(temp_width), math.ceil(temp_height)

def ask_about_picture(base64Str, question):
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question + ". Check the image carefully before answering. Answer shortly and briefly and accurately."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64Str}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300)
        content = response.choices[0].message.content
        return content
    except Exception as e:
        return "I'm sorry, I cannot assist with this request. " + str(e)
        

# Upload the image to the Streamlit app, only PNG
upload_image = st.file_uploader("Upload an image for discussion. You can ask AI anything about the uploaded image.", type=["png", "jpg"])    
if upload_image is not None:
    image = Image.open(upload_image)
    new_width, new_height = calculate_new_dimensions(image.size[0], image.size[1])
    resized_image = image.resize((new_width, new_height))
    base64_image = get_image_base64(resized_image)
    # show the image with new width and height
    st.image(resized_image, width=new_width, caption="Uploaded Image")

# Placeholder for chat messages
chat_container = st.empty()
os.environ['OPENAI_API_KEY'] = consts.API_KEY_OPEN_AI
client = OpenAI()

def show_image_history():
    if len(st.session_state.image_history) == 0:
        st.write("")
        return

    for entry in st.session_state.image_history:
        if len(entry) == 2:  # Check if the entry has exactly two elements
            author, message = entry
            with st.chat_message(author):
                st.write(message)
        else:
            st.error(f"Invalid entry in chat history: {entry}")

# Initialize chat history in session state
if 'image_history' not in st.session_state:
    st.session_state.image_history = []
    
# Chat input for user message
user_message = st.chat_input("Ask something about the uploaded picture...")

if user_message:
    # Add user message to chat history
    st.session_state.image_history.append(('user', user_message))

    # Temporary loading message
    loading_message = "AI is writing..."
    st.session_state.image_history.append(('AI Assistant', loading_message))

    # Display chat history including the loading message
    for author, message in st.session_state.image_history:
        with st.chat_message(author):
            st.write(message)

    # Get AI response
    ai_response = ask_about_picture(base64_image, user_message)

    # Replace the loading message with the actual response
    st.session_state.image_history[-1] = ('AI Assistant', ai_response)

    # Redisplay the chat history with the actual response
    st.experimental_rerun()

show_image_history()  # Show the chat history

# Clear chat history button
if len(st.session_state.image_history) > 0:
    if st.button("Clear Chat History"):
        st.session_state.image_history = []
        show_image_history()  # Show the empty chat history
        chat_container.empty()  # Clear the chat input box
        st.experimental_rerun()

