import streamlit as st
import openai

st.title("Chatting with ChatGPT")
st.sidebar.header("Instructions")
st.sidebar.info(
    '''This is a web application that allows you to interact with 
       the OpenAI API's implementation of the ChatGPT model.
       Enter a **query** in the **text box** and **press enter** to receive 
       a **response** from the ChatGPT
       '''
    )

# Set the model engine and your OpenAI API key
model_engine = "text-davinci-003"
api_key = st.text_input("Enter the Api key.")
api_key = openai.api_key
openai.api_key="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"

def main():
    '''
    This function gets the user input, pass it to ChatGPT function and 
    displays the response
    '''
    # Get user input
    user_query = st.text_input("Enter query here, to exit enter :q", )
    if user_query != ":q" or user_query != "":
        # Pass the query to the ChatGPT function
        response = ChatGPT(user_query)
        return st.write(f"{user_query} {response}")
    
    
def ChatGPT(user_query):
    ''' 
    This function uses the OpenAI API to generate a response to the given 
    user_query using the ChatGPT model
    '''
    # Use the OpenAI API to generate a response
    completion = openai.Completion.create(
                                  engine = model_engine,
                                  prompt = user_query,
                                  max_tokens = 2500,
                                  n = 1,
                                  temperature = 0.5,
                                      )
    response = completion.choices[0].text
    return response
# call the main function
main() 