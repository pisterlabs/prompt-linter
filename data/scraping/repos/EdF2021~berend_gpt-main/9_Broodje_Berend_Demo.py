import os
import openai
import streamlit as st
from PIL import Image

try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
except:
    openai_api_key = st.secrets["OPENAI_API_KEY"]

image = Image.open("berend_gpt/images/broodje_achtergrond.png")

st.set_page_config(
    page_title="Berend-Botje Skills",
    page_icon=" :genie: ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

response = ""
prompt = ""
aprompt = ""
full_response = ""
file_uploaded =""
img_file_buffer = ""

def toggle():
    file_uploaded = img_file_buffer
    return file_uploaded

col1, col2 = st.columns(2)

with col1:
    st.header(" :genie: Berend-Botje Skills")
    st.subheader(
        " :bread: :male-cook: Het broodje van Berend :baguette_bread: \n*waarom zou je moeilijk doen ....?* "
    )
    st.markdown(
        """ ##### Zin in een lekker :sandwich: broodje, gemaakt met de ingredienten die je in :house: hebt? 
    **Berend's Broodje kan je hierbij helpen**
    - Maak een foto van wat je aan eetbare spullen in huis hebt: brood, beleg, ui, eieren, sla, sausjes, ... ja zelfs de inhoud van je koelkast.
    - Upload de foto hier
    - De AI van Berend-Botje verzint dan met deze ingredienten een recept voor je waarmee je het overheerlijke broodje maakt :cooking:
    - Met enig geluk zal de AI zelfs alvast je broodje op de foto zetten. *Met wisselend succes :smiley:* 
    - **:yum: Eet smakelijk!!** """
    )


with col2:
    st.image(
        image,
        caption=None,
        width=240,
        use_column_width=True,
        clamp=True,
        channels="RGB",
        output_format="auto",
    )


uploaded_file = st.file_uploader(
    "**:frame_with_picture: :red[Hier je foto uploaden!]**",
    type=["jpg","png"],
    help="Op dit moment ondersteunen we alleen foto's in jpg, png formaat ",
)


img_file_buffer = st.camera_input("Maak een foto")

if img_file_buffer:
    # To read image file buffer as bytes:
    # bytes_data = img_file_buffer.getvalue()
    # Check the type of bytes_data:
    # Should output: <class 'bytes'>
    # st.write(type(bytes_data))
    with st.spinner('Bezig ingredienten te ontdekken...'):
        
        try:
            import base64
            import requests
    
            # OpenAI API Key
            api_key = os.getenv("OPENAI_API_KEY")
            
            
            # Function to encode the image
            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
            
            # Path to your image
            image_path = img_file_buffer
            
            # Getting the base64 string
            base64_image = base64.b64encode(image_path.read()).decode('utf-8') # encode_image(image_path)
            
            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Welke ingredienten en voedingsprodukten zie je in de image?"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            
            prompt = response.json()["choices"][0]["message"]['content']
            
            print(response.json()["choices"][0]["message"]['content'])
            
        except openai.error.OpenAIError as e:
            print(e.http_status)
            print(e.error)
    
        if not prompt:
            prompt = st.chat_input("Geen foto? Schrijf hier jouw ingredienten")
    

if not uploaded_file:
    if not prompt:
        st.stop()

if uploaded_file:
    try:
        import base64
        import requests

        # OpenAI API Key
        api_key = os.getenv("OPENAI_API_KEY")
        
        
        
        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        
        # Path to your image
        image_path = uploaded_file
        
        # Getting the base64 string
        base64_image = base64.b64encode(image_path.read()).decode('utf-8') # encode_image(image_path)
        
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Welke ingredienten zie je in de image?"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
        
        prompt = response.json()["choices"][0]["message"]['content']
        myimage = Image.open(uploaded_file)
        st.image(
            myimage, 
            caption=None,
            width=240,
            use_column_width=False,
            clamp=True,
            channels="RGB",
            output_format="auto"
        )
        print(response.json()["choices"][0]["message"]['content'])
        # ing_image = Image.open(uploaded_file)
        # response = openai.Image.create_edit(
            # image=open(ing_img, "rb"),
            # prompt="Een broodje met de ingredienten uit {image}",
            # n = 1,
            # size = "512x512"
        #)
        # image_url = response["data"][0]["url"]
        # st.markdown("[Bekijk je broodje](str(response['data'][0]['url']))")
        # st.image(image_url, caption="""### Het heerlijke AI broodje is tot stand gekomen dankzij **powered by OpenAi, ChatGPT en DALE** """, width=340)
        
    except openai.error.OpenAIError as e:
        print(e.http_status)
        print(e.error)
        
if not prompt:
    st.stop()    

openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "system", 
            "content": "Je bent een behulpzame assistent bot, en je kan op basis van ingredienten recepten voor heerlijke broodjes maken."
        }
    )

full_response = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] != "system":
            st.markdown(message["content"])
        
if prompt:
    with st.spinner('Bezig met recept..'):
        pprompt = "Je bent een behulpzame assistent bot, en je kan op basis van ingredienten recepten voor heerlijke broodjes maken. Maak nu een recept voor een heerlijk broodje, gebruik alleen de volgende ingredienten: """ + prompt 
    
    st.session_state.messages.append(
        {
            "role": "user",
            "content": pprompt
        }
    )
    with st.chat_message("user"):
        st.markdown(pprompt)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # full_response = ""
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
    
if full_response == "":
    st.stop()
    
with st.spinner("Bezig met het maken van de afbeelding... "):
        bereidingswijze = str(full_response[:850]) 
        aprompt = (
            """ Maak een foto van een heerlijk broodje  en gebruik hiervoor deze bereidingswijze: """ + bereidingswijze 
           
        )
        myresponse = openai.Moderation.create(
            input=aprompt,
        )
        # st.write(myresponse)

        for i in "  ", "-", "1-9", "\n":
            aprompt = aprompt.replace(i, " ")
        aprompt = aprompt.replace("  ", " ")
        # print(aprompt)

        try:
            
            response = openai.Image.create(prompt=str(aprompt), n=1, size="512x512")
            image_url = response["data"][0]["url"]
            st.markdown("[Bekijk je broodje](str(response['data'][0]['url']))")
            st.image(image_url, caption="""### Het heerlijke AI broodje is tot stand gekomen dankzij **powered by OpenAi, ChatGPT en DALE** """, width=340)
        except openai.error.OpenAIError as e:
            print(e.http_status)
            print(e.error)
