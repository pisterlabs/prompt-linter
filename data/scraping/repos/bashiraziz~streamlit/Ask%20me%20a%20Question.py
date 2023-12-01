import streamlit as st
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

client = OpenAI()
from dotenv import load_dotenv, find_dotenv

# Set your OpenAI API key
_ : bool = load_dotenv(find_dotenv()) # read local .env file
load_dotenv()

def get_openai_response(user_question: str) -> str:
    # Call OpenAI API to get a response
    response: ChatCompletion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",  # Choose an appropriate model
        messages=[
            {
                "content": user_question,
                "role": "user"
            }
        ],
        #id="1",
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

# Streamlit app code
st.header("Provided by  of Bashir Aziz")
st.title("OpenAI Chat App")


# Input field for user to enter a question
user_question = st.text_input("Enter your question:")

# Button to trigger the OpenAI API call and get a response
if st.button("Get Response"):
    if user_question:
        # Call the function to get OpenAI response
        response = get_openai_response(user_question)

        # Display the response
        st.write(f"Answer: {response}")
    
    # Clear the input field after the button is pressed
        user_question = ""  # Set the input field value to an empty string
        
    else:
        st.warning("Please enter a question.")

