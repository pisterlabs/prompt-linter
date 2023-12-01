import os
import openai
import streamlit as st
from PIL import Image

openai_api_key = os.getenv("OPENAI_API_KEY")

image = Image.open('berend_gpt/images/producttoer.jpeg')
st.set_page_config(
        page_title="Berend-Botje Skills",
        page_icon="👋",
        layout="wide",
        initial_sidebar_state="collapsed" )

col1, col2 = st.columns(2)

with col1:
    st.header("Berend-Botje Skills" )
    st.subheader("De ChatGPT kloon\n*waarom zou je moeilijk doen ....?*")
with col2:
   st.image(image, caption=None, use_column_width=True, clamp=True, channels="RGB", output_format="auto")



uploaded_file = st.file_uploader(
    "**HIER KUN JE JOUW PDF, DOCX, OF TXT BESTAND UPLOADEN!!**",
    type=["pdf", "docx", "txt"],
    help="Gescande documenten worden nog niet ondersteund! ",
)


if not uploaded_file:
    st.stop()

try:
    import pandas as pd
    from io import StringIO

    # uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)
# 
    # # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # st.write(stringio)
# 
    # # To read file as string:
        string_data = stringio.read()
        # st.write(string_data)
# 
    # Can be used wherever a "file-like" object is accepted:
        # dataframe = pd.read_csv(uploaded_file)
        # st.write(dataframe)
        ltext = string_data
        print(ltext)

        ltext = ltext.replace("\n", "")
        long_text = ltext       # 'AGI ' * 5000
        ltext = ltext.split()
        t1 = ltext
        totaal = len(t1) 
        derde = int(len(t1)/10)
        deel1 = t1[0:derde]
        deel2 = t1[derde:derde+derde]
        deel3 =  t1[derde+derde:derde+derde+derde]
        woord =" "
        for i in deel1:
            woord += i  + "  "    
        deel1 = woord
        print(deel1)
        woord =" "
        for i in deel2:
            woord+= i  + "  " 
        deel2 = woord
         
        woord =" "
        for i in deel3:
            woord+= i  + "  " 
        deel3 = woord
     
    # file = read_file(uploaded_file)
except Exception as e:
    st.write(e, file_name=uploaded_file.name)
        


# with st.spinner("Indexeren van het document... Dit kan even duren⏳"):
    # chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)
# 
    # if not is_file_valid(file):
        # st.stop()
# 
# 
    # if not is_open_ai_key_valid(openai_api_key, model):
        # st.stop()
# 
# 
   #  
    # folder_index = embed_files(
            # files=[chunked_file],
            # embedding=EMBEDDING if model != "debug" else "debug",
            # vector_store=VECTOR_STORE if model != "debug" else "debug",
            # openai_api_key=openai_api_key,
# 
        # )
    # if uploaded_file:
        # llm2 = get_llm(model=model, openai_api_key=openai_api_key, temperature=0)
        # result = query_folder(
        # folder_index=folder_index,
            # query="Maak een samenvatting van het document dat net is ingelezen. Geef de hoofd thema's aan en bendadruk de belangrijkste onderwerpen. Maak gebruik van het markdown formaat en gebruik hier 5 regels voor. Geef altijd antwoord in HET NEDERLANDS!!",
            # return_all=return_all_chunks,
            # llm=llm2,
            # )
        # st.markdown(" ### Samenvatting")
        # st.markdown(result.answer)


# openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": "Jij bent Berend, een slimme, vriendelijke bot, die heel goed is in notuleren. Je helpt de gebruiker zo goed als je kan wanneer ze vragen of je notulen voor ze wilt maken. Als een vraag niet duidelijk genoeg is, vraag je om meer informatie. Als je desondanks geen antwoord hebt voor de vraag van de gebruiker zeg je 'Sorry, maar ik weet het antwoord niet.'. De notulen die maakt maak je altijd in markdown formaat. Geef altijd antwoord in het Nederlands"})

    
    message_placeholder = st.empty()
    for i in range(1,3):
        if i ==1:
            st.session_state.messages.append({"role": "user", "content": "Hallo Berend. Ik heb een transcript laten inlezen van een overleg. Kan jij die voor mij op hoofdlijnen samenvatten? Dit is deel 1: \n " + str(deel1) })
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
        if i == 2:
            st.session_state.messages.append({"role": "user", "content": "Maak nu kort een samenvating waarbij je de samenvatting van deel 1 gebruikt om samen met deel 2 die ik nu toevoeg een korte samenvatiing maakt van deel 1 en 2 samen. Dit is deel 2: \n " + str(deel2) })
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
        if i == 3:
            st.session_state.messages.append({"role": "user", "content": "Dit is deel 3: \n " + str(deel3) })
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
        


for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            message_placeholder = st.empty()
            if message["role"] != "system":
                st.markdown(message["content"])

if prompt := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "content": prompt})
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
