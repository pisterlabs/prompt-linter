import streamlit as st
from streamlit_elements import elements, mui, html
import time
import replicate
import os
import requests
import openai  # Importing openai module
from PIL import Image
import json

# Basic chatbot logic using predefined Q&A
def basic_bot_response(user_input):
    responses = {
        "hello": "Hello! How can I assist you?",
        "how are you": "I'm just a bot, so I don't have feelings, but thanks for asking! How can I help you?",
        "bye": "Goodbye! If you have more questions, feel free to ask.",
    }
    return responses.get(user_input.lower(), "Sorry, I don't understand that. Please try again.")

def chatbot_page():
    st.title("Basic Chatbot Assistant 🤖")
    st.write("Ask me anything!")

    user_input = st.text_input("You: ", key="userInput")
    if st.button("Send"):
        response = basic_bot_response(user_input)
        st.write(f"Bot: {response}")

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    # Load the GIF from the 'assets' folder with a specified width
    st.image("my_gif.gif", width=300)
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')
os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                        input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                "temperature": 0.1, "top_p": 0.9, "max_length": 512, "repetition_penalty": 1})
    return output

if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate a new text response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)

    # Additional code to check for image-based queries
    # For simplicity, if the user uses the word "image of" in their query, we generate an image
    if "image of" in prompt:
        image_description = prompt.replace("image of", "").strip()
        image_url = generate_image(image_description)
        st.image(image_url, caption=image_description, use_column_width=True)


# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

# Link to the basic chatbot page
if st.sidebar.button("Go to Basic Chatbot"):
    chatbot_page()




def photo_upload_widget():
    """
    Function to handle photo upload widget and its associated operations.
    """
    if "photo" not in st.session_state:
        st.session_state["photo"] = "not_done"

    col1, col2, col3 = st.columns([1, 2, 1])

    # Displaying introduction and info
    col1.markdown("# Welcome to my app!")
    col1.markdown("Here is some info on the app.")

    # Function to change photo state
    def change_photo_state():
        st.session_state["photo"] = "done"

    # Upload or take a photo
    uploaded_photo = col2.file_uploader("Upload a photo", on_change=change_photo_state)
    camera_photo = col2.camera_input("Take a photo", on_change=change_photo_state)

    # Progress bar after photo is chosen
    if st.session_state["photo"] == "done":
        progress_bar = col2.progress(0)
        for perc_completed in range(100):
            time.sleep(0.05)  # Simulating some processing time
            progress_bar.progress(perc_completed + 1)
        col2.success("Photo uploaded successfully!")

    # Displaying a metric
    col3.metric(label="Temperature", value="60 °C", delta="3 °C")

    # Additional info in an expander
    with st.expander("Click to read more"):
        st.write("Hello, here are more details on this topic that you were interested in.")

    # Display the chosen photo
    if uploaded_photo is not None:
        st.image(uploaded_photo, caption="Uploaded Photo", use_column_width=True)
    elif camera_photo is not None:
        st.image(camera_photo, caption="Camera Photo", use_column_width=True)


def streamlit_elements_demo():
    """
    Function to showcase various Streamlit Elements.
    """
    # Displaying Typography
    with elements("new_element"):
        mui.Typography("Hello world with Typography!")

    # Displaying Button with Multiple Children
    with elements("multiple_children"):
        mui.Button(
            mui.icon.EmojiPeople,
            mui.icon.DoubleArrow,
            "Button with multiple children"
        )

    # Nested Children
    with elements("nested_children"):
        with mui.Paper:
            with mui.Typography:
                html.p("Hello world")
                html.p("Goodbye world")

    # Adding Properties to an Element
    with elements("properties"):
        with mui.Paper(elevation=3, variant="outlined", square=True):
            mui.TextField(
                label="My text input",
                defaultValue="Type here",
                variant="outlined",
            )

    # Applying Custom CSS Styles
    with elements("style_mui_sx"):
        mui.Box(
            "Some text in a styled box",
            sx={
                "bgcolor": "background.paper",
                "boxShadow": 1,
                "borderRadius": 2,
                "p": 2,
                "minWidth": 300,
            }
        )

    # Callbacks to Retrieve Element's Data
    with elements("callbacks_retrieve_data"):
        if "my_text" not in st.session_state:
            st.session_state.my_text = ""

        def handle_change(event):
            st.session_state.my_text = event.target.value

        mui.Typography(st.session_state.my_text)
        mui.TextField(label="Input some text here", onChange=handle_change)

def images_api_guide():
    """
    Display the Images API guide with interactive widgets.
    """
    st.title("Images API Guide")

    st.header("Introduction")
    st.write("""
    The Images API provides three methods for interacting with images:
    - Creating images from scratch based on a text prompt
    - Creating edits of an existing image based on a new text prompt
    - Creating variations of an existing image
    This guide covers the basics of using these three API endpoints with useful code samples.
    To see them in action, check out our DALL·E preview app.
    """)

    st.header("Advanced Usage & Tips")
    st.write("""
Here are some advanced features and best practices to make the most out of the Text Image Prompt Generator:

- **Adaptive Prompting**: Our application harnesses the power of GPT-3.5 Turbo, designed to adapt and improve based on your inputs.
- **Exploratory Mode**: Toggle on for unexpected and creative results, perfect for brainstorming and concept development.
- **Session Memory**: Feel free to refine and iterate upon previous prompts within a single session for a more refined output.
- **Safety First**: Our generator has built-in content filters. However, always review prompts especially if they'll be used in public or professional contexts.
- **Seamless Integration**: Use our generated prompts directly with image generators like DALL·E for an end-to-end creative experience.
- **Feedback Loop**: Your feedback helps improve our model. Found an intriguing prompt? Let us know!
- **Performance Tips**: For rapid responses, generate fewer prompts or opt for a smaller model size when possible.
""")

# Generations
st.subheader("Text Image Prompt Generator")
st.write("""
The text image prompt generator endpoint allows you to create a textual prompt for an image based on your input.
The result will be a descriptive text that can be used to generate images.
For instance, you can provide a broad theme, and the generator will give you a more detailed description.
""")

# Interactive Widgets
input_prompt = st.text_input("Enter a broad theme or idea:", "cat")
n_prompts = st.slider("Select number of prompts:", 1, 10, 1)

if st.button("Generate Prompt"):
    # Setting up the headers for the API call
    headers = {
        "Authorization": f"Bearer {st.secrets['openai']['api_key']}",
        "Content-Type": "application/json"
    }

    # API endpoint for chat completions (this is a mock endpoint, replace with the actual endpoint if different)
    endpoint = "https://api.openai.com/v1/chat/completions"

    # Data payload
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": f"Describe an interesting image based on the theme: {input_prompt}"}],
        "temperature": 0.7,
        "max_tokens": 150
    }

    # Making the API call
    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        content = response.json()
        assistant_message = content['choices'][0]['message']['content']
        st.write(f"Generated Prompt: {assistant_message}")
    else:
        st.write(f"Error {response.status_code}: {response.text}")


# Setup
openai.api_key = st.secrets['openai']['api_key']

# Image Generation
def generate_image(prompt, n=1, size="1024x1024"):
    response = openai.Image.create(prompt=prompt, n=n, size=size)
    return response['data'][0]['url']

# Image Edits
def edit_image(image_file, mask_file, prompt, n=1, size="1024x1024"):
    # Assuming openai.Image.create_edit is the correct method to edit images
    response = openai.Image.create_edit(
        image=image_file,  # directly using the UploadedFile object
        mask=mask_file,    # directly using the UploadedFile object
        prompt=prompt,
        n=n,
        size=size
    )
    return response['data'][0]['url']

# Image Generation
def generate_image(prompt, n=1, size="1024x1024"):
    response = openai.Image.create(prompt=prompt, n=n, size=size)
    return response['data'][0]['url']

# Image Edits
def edit_image(image_file, mask_file, prompt, n=1, size="1024x1024"):
    # Convert the image and mask to a supported format (e.g., RGBA)
    image = Image.open(image_file).convert("RGBA")
    mask = Image.open(mask_file).convert("RGBA")

    # Resize the mask to match the image dimensions
    mask = mask.resize(image.size)

    # Save the converted images to temporary files
    image_path_temp = "/tmp/image_temp.png"
    mask_path_temp = "/tmp/mask_temp.png"
    image.save(image_path_temp)
    mask.save(mask_path_temp)

    with open(image_path_temp, "rb") as image, open(mask_path_temp, "rb") as mask:
        # Assuming openai.Image.create_edit is the correct method to edit images
        response = openai.Image.create_edit(
            image=image,
            mask=mask,
            prompt=prompt,
            n=n,
            size=size
        )
    return response['data'][0]['url']

# Image Variations
def create_variation(image_file, n=1, size="1024x1024"):
    # Convert the uploaded file to a supported format (e.g., RGBA)
    image = Image.open(image_file).convert("RGBA")

    # Save the converted image to a temporary file
    image_path_temp = "/tmp/image_variation_temp.png"
    image.save(image_path_temp)

    with open(image_path_temp, "rb") as image:
        response = openai.Image.create_variation(image=image, n=n, size=size)
    return response['data'][0]['url']

# Streamlit UI
st.title("DALL·E-2 Image Generator")

option = st.sidebar.selectbox("Choose an option:", ["Generate Image", "Edit Image", "Create Variation"])

if option == "Generate Image":
    prompt = st.text_input("Enter a prompt for the image:", "a futuristic city skyline")
    if st.button("Generate"):
        image_url = generate_image(prompt)
        st.image(image_url, caption=prompt, use_column_width=True)

elif option == "Edit Image":
    image_file = st.file_uploader("Upload an image:", type=["png"])
    mask_file = st.file_uploader("Upload a mask:", type=["png"])
    prompt = st.text_input("Enter a prompt for the edited image:", "A sunlit indoor lounge area with a pool containing a flamingo")
    if st.button("Edit"):
        image_url = edit_image(image_file, mask_file, prompt)
        st.image(image_url, caption=prompt, use_column_width=True)

elif option == "Create Variation":
    image_path = st.file_uploader("Upload an image for variation:", type=["png"])
    if st.button("Generate Variation"):
        image_url = create_variation(image_path)
        st.image(image_url, use_column_width=True)

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Choose Page", ["Home", "Photo Upload Widget", "Streamlit Elements Demo", "Images API Guide"])

    if selection == "Home":
        st.title("UI & Widgets Playground: Home 🏠")
        st.write("""
    Greetings, curious explorer! 🌟

    Welcome to the central hub of the UI & Widgets Playground. This space has been meticulously crafted to serve as a nexus between creativity and technology, where you're encouraged to play, learn, and innovate.

    ### Why are you here?
    - **Discover the Magic**: Understand the intricacies of Streamlit's diverse range of widgets and how they can elevate your app's user experience.

    - **Craft & Customize**: Learn how to tailor Streamlit's UI components to resonate with your brand's identity and aesthetic.

    - **Interactive Learning**: Don't just read about it; interact with it! Each widget here is fully functional, waiting for your input.

    - **Inspiration Awaits**: Whether you're a seasoned developer or just starting, this playground might just spark that next big idea.

    - **Community & Collaboration**: Remember, you're not alone in this journey. Our vibrant community is always here to support, guide, and collaborate.

    Embark on this journey of exploration and creation, and let's craft web applications that are not only functional but also a delight to interact with. Happy experimenting!
    """)

    elif selection == "Photo Upload Widget":
        st.title("Photo Upload Widget")
        photo_upload_widget()

    elif selection == "Streamlit Elements Demo":
        st.title("Streamlit Elements Demo")
        streamlit_elements_demo()

    elif selection == "Images API Guide":
        images_api_guide()

if __name__ == "__main__":
    main()