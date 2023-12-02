import os
import openai
import streamlit as st
from PIL import Image

openai_api_key = os.getenv("OPENAI_API_KEY")

image = Image.open('images/producttoer.jpeg')
st.set_page_config(
        page_title="Berend-Botje Skills",
        page_icon="👋",
        layout="wide",
        initial_sidebar_state="collapsed" )

response = ""

col1, col2 = st.columns(2)

with col1:
    st.header("Berend-Botje" )
    st.subheader(" :bread: :male-cook: Het broodje van Berend :baguette_bread: :\n*waarom zou je moeilijk doen ....?* ")
    st.markdown(""" ##### Heb je rond deze tijd ook zo'n trek in een lekker broodje :sandwich: maar je hebt geen zin om de deur uit te gaan :house: ? 
    **Dan heeft Berend's Broodje de oplossing.**
    - Stuur een foto van wat je in huis hebt: brood, beleg, sla, sausjes, ... wat je maar wilt 
    - Berend maakt dan een recept voor je om snel een heerlijk broodje te maken :cooking:  
    - Hij stuurt zelfs een foto mee, om je alvast lekker te maken

    **Eet smakelijk!!** """ )


with col2:
   st.image(image, caption=None, width=240, use_column_width=True, clamp=True, channels="RGB", output_format="auto")


uploaded_file = st.file_uploader(
        "**:notebook: Hier je foto uploaden!**",
        type=["jpg"],
        help="Gescande documenten worden nog niet ondersteund! ",
)


# openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": "Geef altijd antwoord in het Nederlands"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] != "system":
            st.markdown(message["content"])


full_response =""

if prompt := st.chat_input("Hoe gaat het?"):
    st.session_state.messages.append({"role": "user", "content": "Geef een recept voor een lekker broodje als ik deze spullen in huis heb:  " + prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
            
    
    message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    response ==" " 
    
    print(full_response)
    
    
    if full_response == "":
        st.stop()
    

    try:
        with st.spinner("Bezig met het maken van de afbeelding... "):

            aprompt = """ Maak een foto van een broodje volgens dit recept """ + ' """ ' + str(full_response[0:700]) + ' """ '        
            myresponse = openai.Moderation.create(
                    input = aprompt,
                 )
            print(myresponse)
            
            for i in "  ", "-", "1-9", "\n":
                aprompt = aprompt.replace(i," ")

            aprompt = aprompt.replace("  ", " ")
            print(aprompt)


            response = openai.Image.create( prompt=str(aprompt), n=1, size="1024x1024")        
            image_url = response['data'][0]['url']
        # image = Image.open(image_url)
        # st.image(image, caption="Heerlijk broodje met jouw ingredienten", width=256, use_column_width=False, clamp=True, channels="RGB", output_format="auto")
        # st.markdown("[Heerlijk broodje](" + str(image_url) + ")")
        
        st.image(image_url, caption=" ### Het heerlijke broodje is tot stand gekomen dankzij **powered by OpenAi, ChatGPT en DALE**", width=340 )
        # print(response['data'][0]['url'])
    except openai.error.OpenAIError as e:
        print(e.http_status)
        print(e.error)
    
    
    # st.write(image_url)
