import streamlit as st
import os
import openai
import time
from PIL import Image 
from streamlit_extras.stoggle import stoggle
from streamlit_extras.streaming_write import write
from dotenv import load_dotenv

# OpenAI API key setup start----

load_dotenv()

api_key = os.getenv("2OPENAI_API_KEY")

if api_key is None:
    raise Exception("API key not found in .env file")

openai.api_key = api_key
# OpenAI API key setup finish----

# -----------------------------------


# Function to communicate with ChatGPT
def chat_with_gpt(prompt):
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
        "role": "system",
        "content": "You are my personal assistant. Your name is MealMaster. You are great at explaining and helping me and love what you do."
            },
            {
        "role": "user",
        "content": prompt
            }
            ],  
        max_tokens=1000  # Adjust the max_tokens as needed
    )
            

    return response['choices'][0]['message']['content']




#  ------------------------------------------------------------



col1, col2 = st.columns((2,1))
with col1:
    mtitle = ('## Welcome to MealMaster! 🥗')
    subhead = ("Recipes in seconds.")
    def stream_example():
        for word in mtitle.split():
            yield word + " "
            time.sleep(0.23)

    def stream_2():
         for word in subhead.split():
            yield word + " "
            time.sleep(0.13)   
    write(stream_example)
    write(stream_2)
    st.divider()
    st.markdown(
        """      
        Features included:
        - Can add allergies / dietary restrictions
        - Can input just a name like "pasta" to get you started
        - Can input your own ingredients and have a recipe generated
        """
    )

signuptitle = st.empty()
with signuptitle:
    st.subheader(
    """Click the link below to start using today!
    """)

signuplink = st.empty()
with signuplink:    
    st.write(
    """https://buy.stripe.com/cN2eVu4CG0rg8Lu4gg
    """)

faq = st.empty()

with faq.expander("Tips for use/FAQ's:"):
        st.write('''
           - Once you subscribe to MealMaster, you will see a thank you message on your screen with your password.
            This is the password you will enter in the password field below.
        ''')
        st.write('''
           - Press the button ONCE and let the recipe generate. It may be slower than you're used to but your recipe is coming.
        ''')
        st.write('''
            - If you click the button more than once, it will restart your time.     
        ''')

# 48fnsl489dj
# https://buy.stripe.com/cN2eVu4CG0rg8Lu4gg
# 4b3a9z

# password = "4b3a9z"

# with col2:
#     image = Image.open('cutlery-knife-svgrepo-com.png')
#     st.image(image)


logintitle = st.empty()

with logintitle:
    st.subheader(
    """Already signed up? Sign in below:
    """,
    )

login = st.empty()

with login.form("login_form"):

    st.write("Login")
    email = st.text_input('Enter Your Email')
    password = st.text_input('Enter Your Password')
    submitted = st.form_submit_button("Login")


if submitted:
    if password == "4b3a9z":
        st.session_state['logged_in'] = True
        st.text('Succesfully Logged In!')
    if password == "nkosua":
        st.session_state['logged_in'] = True
        st.text('Succesfully Logged In!')
    else:
        st.text('Incorrect, login credentials.')
        st.session_state['logged_in'] = False

#  ---------------------------------------------------

# Choose input type

with st.sidebar:
    st.header("Instructions")

    st.write('''  
    - For Ingredients mode: Provide what ingredients you would like a recipe with in the text field.
    ''')
    st.write('''  
    - For the Dish mode: Provide a name of a dish you would like to make MealMaker will come up with a recipe for you. If you have any allergies, dont forget to put them in the allergy text field. Then click 'Cook me a meal!' to generate your recipe.
    ''')
    st.caption("Brought to you by Itwela Ibomu")
    






#  ----------------------------------------------------

if 'logged_in' in st.session_state.keys():
    if st.session_state['logged_in']:
        login.empty()
        logintitle.empty()
        faq.empty()
        signuptitle.empty()
        signuplink.empty()
        st.title("MealMaster:")
        stoggle(
        "Instructions:",
        """
            For Ingredients mode: Provide what ingredients you would like a recipe with in the text field.
            For the Dish mode: Provide a name of a dish you would like to make MealMaker will come up with a recipe for you. If you have any allergies, dont forget to put them in the allergy text field. Then click 'Cook me a meal!' to generate your recipe.
        """,
        ) 
        st.divider()         
        st.write('''Please choose a mode below:''')
        input_type = st.selectbox("Choose a mode:", ["Ingredients", "Dish"])
        # Get user input
    if input_type == "Ingredients" :
        allergies = st.text_input("Any Allergies? If not you can leave blank:")
        ingredients = st.text_input("Enter ingredients separated by commas:")
        prompt = f"Create a recipe using the following ingredients: {ingredients}. Provide a recipe name, ingredients and detailed steps. If I have any allergies, I will input them here: {allergies}. If this is blank, you can ignore the allergies all together. Add calories for the recipe as well"
    else:
        allergies = st.text_input("Any Allergies? If not you can leave blank:")
        recipe_name = st.text_input("Enter the dish name:")
        prompt = f"Provide ingredients and detailed steps for the following recipe: {recipe_name}. Provide a recipe name, ingredients and detailed steps. If I have any allergies, I will input them here: {allergies}. If this is blank, you can ignore the allergies all together Add calories for the recipe as well."


    recipe_response = ""

    # Send query to the chatbot
    if st.button("Cook me a meal!"):
        pbtext = ("Please wait, your recipe will load shortly......")
        prg = st.progress(0, text=pbtext)
  

        recipe_response = chat_with_gpt(prompt)
        
        # Split the response into lines and find the recipe name
        lines = recipe_response.split('\n')
        recipe_name = ""
        ingredients_and_steps = ""
        for line in lines:
            if "recipe name" in line.lower():
                recipe_name = line.strip()
            else:
                ingredients_and_steps += line.strip() + "\n"
        
        for i in range(99):
            time.sleep(0.2)
            prg.progress(i+1, text=pbtext)
        
        for i in range(1):
            time.sleep(0.2)
            prg.progress(100, text="All done!")

        
        # Output
        with st.container():
            st.markdown(f"## {recipe_name}")
            st.markdown(ingredients_and_steps)

#  ----------------------------------------

