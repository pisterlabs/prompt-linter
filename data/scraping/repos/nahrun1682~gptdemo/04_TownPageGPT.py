import os
import sys
import datetime
import openai
import streamlit as st
from dotenv import load_dotenv

from audio_recorder_streamlit import audio_recorder

from libs.get_keywords import return_keywords_for_google
from libs.get_townpage_by_google import get_Townpage_by_google

import shutil

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# working_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(working_dir)


st.sidebar.title("Secretary GPT")


# Configurar la clave de la API de OpenAI
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key= os.environ["OPENAI_API_KEY"]

def transcribe(audio_file):
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript


def summarize(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=(
            f"Please do what you are asked to do with the following text:\n"
            f"{text}"
        ),
        temperature=0.5,
        max_tokens=560,
    )

    return response.choices[0].text.strip()

st.image("https://hablemosbien.org/wp-content/uploads/2023/08/bot-e1692659787339.png") 

st.write("Click on the microphone and tell your GPT secretary what to type.")
st.write("Ex.: 'Write an email to Mary asking for the financial report.'")

# Explanation of the app
st.sidebar.markdown("""
## Instructions
1. Enter your OpenAI api key.
2. Make sure your browser allows microphone access to this site.
3. Choose to record audio or upload an audio file, in any major language.
4. To begin, tell your secretary what to record: an email, a report, an article, an essay, etc.
5. If you record audio, click on the microphone icon to start and to finish.
6. If you are uploading an audio file, select the file from your device and upload it.
7. Maximum recording time: 5 minutes.
8. Click on Transcribe. The waiting time is proportional to the recording time.
9. The transcription appears first and then the request.
10. Download the generated document in text format.
-  By Moris Polanco
        """)

# 指定したディレクトリのパスを設定
audio_dir = "gptdemo/tmp/Audio"

# ディレクトリが存在しない場合は作成
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir, exist_ok=True)

# tab record audio and upload audio
tab1, tab2 = st.tabs(["Upload Audio","Record Audio"])

with tab1:
    audio_file = st.file_uploader("Upload Audio", type=["mp3", "mp4", "wav", "m4a"])

    if audio_file:
        # st.audio(audio_file.read(), format={audio_file.type})
        # Reset file pointer to the beginning
        audio_file.seek(0)
        timestamp = timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # save audio file with correct extension
        ###元のコード
        # with open(f"audio_{timestamp}.{audio_file.type.split('/')[1]}", "wb") as f:
        #     f.write(audio_file.read())
        # 拡張子を取得
        file_extension = audio_file.type.split('/')[1]
        # ファイルパスを指定ディレクトリに設定
        audio_file_path = os.path.join(audio_dir, f"audio_{timestamp}.{file_extension}")
        ###元のコード
        with open(audio_file_path, "wb") as f:
            f.write(audio_file.read())

        #st.audio(audio_file_path, format=f"audio/{file_extension}")
with tab2:
    audio_bytes = audio_recorder(pause_threshold=300)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # save audio file to mp3
        with open(f"audio_{timestamp}.mp3", "wb") as f:
            f.write(audio_bytes)



if st.button("Transcribe"):
    # check if audio file exists
    if not any(f.startswith("audio") for f in os.listdir(".")):
        st.warning("Please record or upload an audio file first.")
    else:
        # find newest audio file
        audio_file_path = max(
            [f for f in os.listdir(".") if f.startswith("audio")],
            key=os.path.getctime,
    )
        

    # transcribe
    audio_file = open(audio_file_path, "rb")

    transcript = transcribe(audio_file)
    text = transcript["text"]

    st.subheader("Transcript")
    st.write(text)

    # summarize
    # summary = summarize(text)

    # st.subheader("Document")
    # st.write(summary)

    # save transcript and summary to text files
    with open("transcript.txt", "w") as f:
        f.write(text)

    # with open("summary.txt", "w") as f:
    #     f.write(summary)
    
    # download transcript and summary
    st.download_button('Download Document', text)
    
    # set a session state variable to hold the transcript
    st.session_state.transcript = text
    
    # remove the audio file after processing
    #os.remove(audio_dir)
    # ディレクトリ内のすべてのファイルを削除
    for filename in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    
# Keywords Extraction Button
if st.button("キーワード抽出"):
    # check if the transcript is available in the session state
    if 'transcript' in st.session_state and st.session_state.transcript:
        # Extract keywords for google search
        keywords = return_keywords_for_google(st.session_state.transcript)
        keywords_list = keywords[0].split(",")
        keywords_list = [keyword.replace('"', '') for keyword in keywords_list]
        st.subheader("Keywords for Google Search")
        st.write(keywords)
        print(keywords_list)
        print(type(keywords_list))
        print(keywords_list[0])
        print(keywords_list[1])
        
        st.session_state.keywords_list = keywords_list
    else:
        st.warning("Please transcribe the audio file first.")
        
if st.button("googleで検索"):
    # check if the transcript is available in the session state
    if 'keywords_list' in st.session_state and st.session_state.keywords_list:
        # Extract keywords for google search
        print("st.session"+st.session_state.keywords_list[0])
        ans = get_Townpage_by_google(st.session_state.keywords_list)
        st.subheader("answer")
        st.write(ans)
    else:
        st.warning("Please transcribe the audio file first.")

# delete audio and text files when leaving app
if not st.session_state.get('cleaned_up'):
    files = [f for f in os.listdir(".") if f.startswith("audio") or f.endswith(".txt")]
    for file in files:
        os.remove(file)
    st.session_state['cleaned_up'] = True