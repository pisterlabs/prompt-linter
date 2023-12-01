'''This chatbot uses the OpenAI ChatGPT API to correct German grammar in the responses to user's prompts.'''
import streamlit as st
import openai
import time

openai.api_key = st.secrets["OPENAI_API_KEY"]

# App title
st.set_page_config(page_title="💬 Grammatik ChatGPT")

"st.session_state:", st.session_state

# Choose OpenAI model
if "openai_model" not in st.session_state.keys():
    st.session_state.openai_model = "gpt-3.5-turbo"

# Initialize chat messages
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hallo! Wie war dein Tag heute?"}]

# Display chat messages
for message in st.session_state.messages:
    if st.chat_message != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

if prompt := st.chat_input():
    # Prepend the system message to the list of messages
    st.session_state.messages.insert(0, {"role": "system", "content": "Du bist ein erfahrener Deutschlehrer \
    und hast eine Konversationsstunde mit einem Schüler. \
    Du verwendest eine einfache und verständliche Formulierung und beginnst damit, \
    den vom Schüler geschriebenen Satz zu analysieren und auf Fehler in Rechtschreibung und Grammatik hinzuweisen. \
    Du erklärst Fehler unter Berücksichtigung der grammatikalischen Regeln, \
    wie z. B. die Nichtübereinstimmung von Genus, Numerus, Kasus, Tempus und Modus von Wörtern, die Verwendung von Präpositionen usw. \
    Ignoriere Probleme mit der Zeichensetzung. \
    Wenn der Satz ungewöhnlich ist, formuliere ihn so um, dass er sich idiomatischer anhört, \
    als wenn ein deutscher Muttersprachler ihn sagen würde. Leite den umformulierten Satz mit So würde ich es sagen: ein. \
    Gib dann eine Antwort, wenn dir eine Frage gestellt wurde oder einen Kommentar. \
    Beantworte jede Korrektur mit einer Frage, um den Schüler zu motivieren. \
    Die Folgefrage sollte sich nicht auf die Grammatik beziehen, sondern versuchen, herauszufinden, was den Schüler interessiert."})

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
