import streamlit as st
import tensorflow as tf
import numpy as np
import openai
import re
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.prompts import PromptTemplate, StringPromptTemplate
from PIL import Image
import os



# Input fields for OpenAI API key and Wolfram Alpha API key
api_key = st.sidebar.text_input('Enter your OpenAI API key', type="password")
wolfram_key = st.sidebar.text_input('Enter your Wolfram Alpha API key',type="password")

# Initialize a warning message
warning_message = ""

# Check if both API keys are not provided
if not api_key and not wolfram_key:
    warning_message = "Please enter both your OpenAI API key and Wolfram Alpha API key."

# Check if OpenAI API key is provided but Wolfram Alpha API key is missing
elif api_key and not wolfram_key:
    warning_message = "Please enter your Wolfram Alpha API key."

# Check if Wolfram Alpha API key is provided but OpenAI API key is missing
elif not api_key and wolfram_key:
    warning_message = "Please enter your OpenAI API key."

# Display the warning message, if any
if warning_message:
    st.sidebar.warning(warning_message)
else:
    # Both keys provided, store them in environment variables
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key
    os.environ["WOLFRAM_ALPHA_APPID"] = wolfram_key
# Initialize conversation memory
conversation_memory = []

prediction=""

# Define class labels for each model
class_labels = {
    'Potato Model': ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'],
    'Pepper Model': ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy'],
    'Tomato Model': [
        'Tomato_Bacterial_spot',
        'Tomato_Early_blight',
        'Tomato_Late_blight',
        'Tomato_Leaf_Mold',
        'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite',
        'Tomato__Target_Spot',
        'Tomato__Tomato_YellowLeaf__Curl_Virus',
        'Tomato__Tomato_mosaic_virus',
        'Tomato_healthy'
    ]
}

# Define model paths
model_paths = {
    'Potato Model': 'potato.tflite',
    'Pepper Model': 'pepper.tflite',
    'Tomato Model': 'tomato.tflite'
}

# Helper function for model inference
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image)
    image = (image.astype('float32') / 255.0)
    return image

def predict(image, model, class_labels):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)

    model.set_tensor(model.get_input_details()[0]['index'], image)
    model.invoke()
    output = model.get_tensor(model.get_output_details()[0]['index'])
    predicted_class = class_labels[np.argmax(output)]
    return predicted_class

# Set your OpenAI API key
openai.api_key = api_key


# Create two columns
col1, col2 = st.columns(2)

# Add content to the first column
with col1:
    st.title('CROPGUARD : Your Friendly Neighbourhood Plant Disease Detector ')
# Add content to the second column
with col2:
    st.image("855c4f3b09f2454eaebbb3baacb982b2.gif", use_column_width=True)
st.session_state.sidebar_state = 'expanded'
# Streamlit app

st.warning("Note: Only diseases listed in the class labels can be predicted, i.e. Potato Early Blight, Potato Late Blight, Bell Pepper bacterial spot, Tomato Bacterial Spot, Tomato Early Blight, Tomato late Blight, Tomato Leaf Mold, Tomato Septorial Leaf Spot, Tomato Spider Mites Two spotted spider mites, Tomato target spot, Tomato yellow leaf curl virus, Tomato Mosaic virus.")



# User selects the model
selected_model = st.selectbox('Select a Model', list(model_paths.keys()))
st.session_state.selected_model = selected_model

# Load the chosen model
model_path = model_paths[selected_model]
model = tf.lite.Interpreter(model_path=model_path)
model.allocate_tensors()

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    image = Image.open(uploaded_image)
    predicted_class = predict(image, model, class_labels[selected_model])
    prediction=predicted_class
    if st.button('Detect'):
        # Set the sidebar visibility to True when "Detect" is clicked
        st.session_state.sidebar_state = 'expanded'
        st.sidebar.title('Results')
        st.sidebar.write(f'Predicted Class ({selected_model}): {predicted_class}')

# Initialize conversation history file path
conversation_history_file = "conversation_history.txt"

template = """Given the plant disease , greet them first and then write the cause , cure and symptoms of the plant diease.
speak in a friendly ay without uneccesary technical words.
Your response:"""
prompt = PromptTemplate.from_template(template)

# State for controlling conversation
if st.checkbox("Start Conversation"):
    st.session_state.sidebar_state = 'expanded'# Initialize detected disease variable

    st.sidebar.write(f"Predicted Disease for the uploaded plant : {prediction}")
    with st.form(key='conversation_form'):
        user_question = st.text_input("Ask a question about the disease or crop (type 'Disease {predicted_disease}'):", key="my")
        submit_button = st.form_submit_button("Submit")

        # Define ChatOpenAI agent for question-answering
        llm = ChatOpenAI(openai_api_key=openai.api_key, temperature=0.7, prompt=prompt)  # Adjust the temperature as needed
        tools = load_tools(['wikipedia', 'wolfram-alpha'], llm=llm)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Define Agent
        agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                                 verbose=True, memory=memory, max_iterations=6 )

        if user_question:
            try:
                if submit_button:
                # Retrieve answer using ChatOpenAI based on the conversation history
                    
                    # Create a list of messages for the conversation
                    messages = [ {"role": "system", "content": "You are a conversational agent. You give information on the cause , cure and symptoms of the plant diease."}, {"role": "user", "content": user_question}
                    ]

                    try:
                        # Use OpenAI's Chat API to get a response
                        response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages
                        )

                        # Extract the assistant's reply
                        assistant_response = response['choices'][0]['message']['content']

                        # Print the assistant's response
                        st.write("CropGuard :")
                        st.write(assistant_response)
                        print(assistant_response)

                        # Save the conversation history
                        conversation_memory.append({"role": "user", "content": user_question})
                        conversation_memory.append({"role": "assistant", "content": assistant_response})

                                # Check if the conversation history file exists
                        if os.path.exists(conversation_history_file):
                        # Append the conversation to the existing file
                            with open(conversation_history_file, "a") as file:
                                for item in conversation_memory:
                                    file.write(f"{item['role']}: {item['content']}\n")
                        else:
                        # Create a new file and write the conversation history to it
                            with open(conversation_history_file, "w") as file:
                                for item in conversation_memory:
                                    file.write(f"{item['role']}: {item['content']}\n")


                    except Exception as e:
                        # Handle exceptions
                        if "Could not parse LLM output:" in str(e):
                            info = str(e).removeprefix("Could not parse LLM output: `").removesuffix("`")
                        else:
                            raise Exception(str(e))
                        st.error(f"An error occurred: {info}")


            except Exception as e:
                raise Exception(str(e))

with st.sidebar.expander("About CropGuard"):
        st.title("About CropGuard")
        st.write("CropGuard is your friendly neighborhood plant disease detector.")
        st.write("It uses machine learning models to identify diseases in plants, such as potatoes, peppers, and tomatoes.")
        st.write("Simply upload an image of a plant, and CropGuard will predict the disease it might have.")
        st.write("Additionally, you can have a conversation with CropGuard to learn more about the disease, its causes, cures, and symptoms.")
        st.write("CropGuard is designed to provide information in a friendly and understandable way.")
        st.write("If you have any questions or feedback, please feel free to reach out to us.")
        st.write("Thank you for using CropGuard!")

# Add a button to download the conversation history file
# Add a button to download the conversation history file
if st.sidebar.button("Download Conversation History"):
    with open(conversation_history_file, "r") as file:
        history_text = file.read()
    st.download_button(
        label="Click to Download Conversation History",
        data=history_text,
        key="download_conversation_history",
        file_name="conversation_history.txt",
    )

    

