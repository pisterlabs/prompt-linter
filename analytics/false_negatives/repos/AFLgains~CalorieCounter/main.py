import os
import streamlit as st
import streamlit_authenticator as stauth
from openai import OpenAI
from dotenv import load_dotenv
import base64
from PIL import Image
from io import BytesIO
from prompts import SYSTEM_PROMPT
from azure.data.tables import TableClient

load_dotenv()

def encode_image(image_file):
    img = Image.open(image_file)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def calc_costs(input,output):
    return input*0.03/1000 + output*0.06/1000

if 'register_visible' not in st.session_state:
    st.session_state['register_visible'] = True

if 'disable_picture' not in st.session_state:
    st.session_state['disable_picture'] = True


# TODO: Ensure this is loadedfrom a database instead
connection_string = os.environ["CONNECTION_STRING"]
table_client  = TableClient.from_connection_string(conn_str=connection_string,table_name = "userdetails")
user_entities = table_client.list_entities()

def make_credentials(user_entities):
    credentials = {"usernames":{e['username']:{
        "email":e['email'],
        "name":e['PartitionKey'],
        "password":e['hashed_password']} for e in user_entities}}
    return credentials

#Configure the authenticator
authenticator = stauth.Authenticate(
    make_credentials(user_entities),
    os.environ["COOKIE_NAME"],
    os.environ["COOKIE_KEY"],
    int(os.environ["EXPIRY"]),
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    st.session_state['register_visible'] = False
    authenticator.logout('Logout', 'main')
    st.markdown(f"<h1 style='text-align: center; color: black;'>Welcome {name}</h1>", unsafe_allow_html=True)
    st.write('Welcome to calorie counter! Upload a picture of your food to get an estimated caloric value and macro nutrient profile!')
    
    picture = st.camera_input("Take a picture")
    if picture:
        st.image(picture)
    
        details = st.text_input("Any more details you'd like to add? E.g., dish type, serving size etc. Details improves the estimate, but aren't needed")
        if details == "":
            details = "None"
        calculate = st.button("Calculate!")
        if calculate:
            with st.spinner('Analysing your food...'):
                client = OpenAI(api_key = os.environ["OPENAI_API_KEY_GPT"])
                response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        }
                        ,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", 
                                "text": "Additional Details: {details}. Estimate the calories and macronutrient breakdown. "},
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/jpeg;base64,{encode_image(picture)}",
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )

            st.write(response.choices[0].message.content)

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

if st.session_state['register_visible']:
    signuplink = os.environ["SIGNUP_LINK"]
    st.markdown(f"Not registered? Sign up [here]({signuplink})")