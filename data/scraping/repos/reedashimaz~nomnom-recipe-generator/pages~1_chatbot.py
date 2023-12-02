from openai import OpenAI
import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv

st.set_page_config(
    page_title="Chatbot",
    layout='wide'
)

if 'total_groceries' not in st.session_state:
    st.session_state.total_groceries = {}

with open("total_groceries.txt", "r") as file:
    st.session_state.total_groceries = eval(file.read())

# Load environment variables
_ = load_dotenv(find_dotenv())

# Get OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Google Drive file ID for the video
video_file_id = '1miZ6Zz4lt0wxTGbrtR8IltVr_fOPX6Vr'
video_url = f'https://drive.google.com/uc?export=view&id={video_file_id}'

# Create columns for layout
col1, col2 = st.columns([0.20, 0.80])

# Column 1: Display the embedded video without controls
with col1:
    video_html = f"""
    <video width="100%" height="auto" autoplay muted style="margin-top: -30px;">
      <source src="{video_url}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """
    st.markdown(video_html, unsafe_allow_html=True)

with col2:
    st.title("NomNom Bot")
    st.markdown('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.caption("🍳 Your personal recipe generator!")
st.caption("Enter a cuisine and your dietary restrictions to get started :)")

with st.sidebar:
    cuisine = st.text_input("Cuisine you're craving", key="cuisine")
    diet_res = st.text_input("Dietary Restrictions", key="diet_res")

ingredients = {
    'Tomatoes': 3,
    'Rice': 1
}
ingredients = st.session_state.total_groceries

content_template = """
I have scanned/entered my grocery store receipts. Here are the items and their quantity (in the form of a dictionary) I have purchased: 
{ingredients}.
My dietary restrictions include {diet_res}. I am in the mood for {cuisine} recipes. Give me a detailed recipe using ONLY
the ingredients I have. I also have common pantry items like salt, pepper, olive oil, and basic spices. 
Make sure to take the cuisine and dietary restrictions into consideration definitively. 
Provide a recipe that aligns with the preferred cuisine given by the user. A cuisine is a style or method of cooking, especially of a particular country
or region. Use ingredients, foods, and techniques from the cuisine from that specific region.
If the user says their dietary restriction is an X allergy or allergic to X where X is any ingredient, make sure to exclude X from the recipe
since the user could die of an allergic reaction and you will be responsible.
Start your response with "Howdy, I'm the NomNom Bot!" and end by asking if the user has any follow-up questions or needs any additional resources.
"""
print("Content Template:", content_template.format(ingredients=ingredients, diet_res=diet_res, cuisine=cuisine))
# Initialize the chat with an initial user message
if cuisine and diet_res:
    if "messages" not in st.session_state:
        print("Content Template:", content_template.format(ingredients=ingredients, diet_res=diet_res, cuisine=cuisine))
        st.session_state["messages"] = [
            {"role": "user", "content": content_template.format(ingredients=ingredients, diet_res=diet_res, cuisine=cuisine)}]

    # Display the chat messages
    for msg in st.session_state.messages[1:]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Generate a response for the initial user message
    if len(st.session_state.messages) == 1:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.5
        )

        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

    # User can ask follow-up questions
    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("API key is not valid.")
            st.stop()

        client = OpenAI(api_key=openai_api_key)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.5
        )

        msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)