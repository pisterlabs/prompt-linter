import openai
import streamlit as st
import os
from elevenlabs import generate, play, stream, set_api_key, VoiceSettings, Voice, voices
from elevenlabs.api.error import UnauthenticatedRateLimitError, RateLimitError


def autoplay_audio(file_path: str):
    import base64
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def speak(text, voice):
    try:
        audio = generate(text=text, 
                         voice=Voice(
                                voice_id=voice,
                                settings=VoiceSettings(stability=0.71, similarity_boost=1, style=0.0, use_speaker_boost=True)
                            ),
                         model='eleven_multilingual_v2'
                         )
        # byteをファイルに書き込む
        audio_path = "audio.mp3"
        with open(audio_path, mode="wb") as f:
            f.write(audio) # type: ignore

        autoplay_audio(audio_path)
    except UnauthenticatedRateLimitError:
        e = UnauthenticatedRateLimitError("Unauthenticated Rate Limit Error")
        st.exception(e)

    except RateLimitError:
        e = RateLimitError('Rate Limit')
        st.exception(e)

st.title("LangCore Text2Speech")

openai.api_base = "https://oai.langcore.org/v1"
elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
if elevenlabs_api_key:
    set_api_key(elevenlabs_api_key)
else :
    st.error("ELEVENLABS_API_KEY is not set")

# ここどう書くのがいいんだろ
voice_list = voices()
voice_id_list = [voice.voice_id for voice in voice_list]
voice_name_list = [voice.name for voice in voice_list]
marin_index = voice_name_list.index("Kitahara")
selected_voice_name = st.selectbox("Select voice", options=voice_name_list, index=marin_index)
selected_voice_id = voice_id_list[voice_name_list.index(selected_voice_name)]
st.write(selected_voice_id)

with st.expander("Click to expand and enter system prompt"):
    system_prompt = st.text_area("Enter system prompt", value=f"""あなたは {selected_voice_name} です。
ユーザに回答する際になるべく短く回答するようにしてください。目安は10文字から20文字です。""")
    
    embeddings_group_name = st.text_input("Enter embeddings group name(optional)", value="")
    if embeddings_group_name:
        system_prompt += """
# You can use below knowledge to respond to User's statement. 必要に応じて以下の知識を使ってユーザに話題を提供してください。

{{EMBEDDINGS_CONTEXT}}
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Chat with my voice")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            query=prompt,
            groupName = embeddings_group_name,
            headers = {
                "Content-Type": "application/json",
                "LangCore-Embeddings": "on",
                "LangCore-Embeddings-Match-Threshold": "0.8",
                "LangCore-Embeddings-Match-Count": "3",
            } ,
            messages= [
                {
                    "role": "system",
                    "content": system_prompt
                },
                *st.session_state.messages
                ],
                stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "") # type: ignore
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    with st.spinner('Generating audio...'):
        speak(full_response, selected_voice_id)
