import openai
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

file1 = open("moods.txt","r")
file2 = open("type.txt","r")


st.set_page_config(page_title="Heading | Automated", page_icon="🤖")


st.title("The Heading for All")
st.sidebar.header("Instructions")
st.sidebar.info(
    '''This is a web application that allows you to generate  for your projects and work! You can describe **The situation for which you are generating** and choose the **tone** 
        for the heading. The caption will be generated using AI! 
    '''
    )
st.sidebar.info("Pro tip: Make your description specific for best results.")
st.sidebar.info("Pro tip: Use the slider to adjust the creativity of the caption.")
st.markdown(
    "This mini-app generates Heading such as Titles, Slogans, Captions, Subjects using OpenAI's GPT-3 based [Davinci model](https://beta.openai.com/docs/models/overview). You can find the code on [GitHub](https://github.com/adarshxs/Instagram-Automation) and the author on [Linkedin](https://www.linkedin.com/in/adarsh-a-s/)."
)


model_engine = "text-davinci-003"
openai.api_key = os.getenv("api_key")

def main():

    # Get user input
    param1=st.selectbox("Type",(file2.readlines()))
    param=st.selectbox("Tone", (file1.readlines()))
    temp = st.slider("Creativity", 0.0, 1.0, 0.50)
    st.subheader('Generated Heading :sunglasses:')
    user_query = st.text_input("Briefly describe your Situation and reason you are writing for")


    if st.button("Generate"):
        if user_query != ":q" or user_query != "":
            # Pass the query to the ChatGPT function
            response = ChatGPT(user_query, temp, param,param1)
            return st.code(response, language='None')
        
    
    
def ChatGPT(user_query, temp, param,param1):
    # Use the OpenAI API to generate a response
    completion = openai.Completion.create(
                                engine = model_engine,
                                prompt = "Write a"+ param + param1 +"about" + user_query + ".Also it is for projects and fun use both",
                                max_tokens = 100,
                                n = 1,
                                temperature = temp,
                                
                                )
    response = completion.choices[0].text
    response=response.replace('"','')
    return response
main()
st.markdown("""---""")
# col1, col2 = st.columns(2)
# with col1:
#     st.markdown()

#LOGIN PAGE

# d1={"Nimish":"hello"}

# def login_page():
#     st.title("Login Page")
#     name = st.text_input("Enter name", "")
#     passw = st.text_input("Enter Password", type="password")

#     if st.button("Sign In"):
#         if name in d1 and d1[name] == passw:
#             login_placeholder.empty()  # Remove login page content
#             home_page()  # Render home page
#         else:
#             st.write("Wrong username or password")

# login_placeholder = st.empty()  # Create a placeholder for the login page
# login_page()
