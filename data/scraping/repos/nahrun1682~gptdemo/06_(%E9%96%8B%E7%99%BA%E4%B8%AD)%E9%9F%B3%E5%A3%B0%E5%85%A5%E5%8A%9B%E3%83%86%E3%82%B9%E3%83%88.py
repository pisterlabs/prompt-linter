import os
import sys
import datetime
import openai
import streamlit as st
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
from st_audiorec import st_audiorec

# st.title('🦜ChatGPT')
# st.subheader("TownPageGPT(デモ)")


# working_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(working_dir)

# .envファイルの読み込み
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))


# Configurar la clave de la API de OpenAI
api_key = os.environ["OPENAI_API_KEY"]

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')

# DESIGN implement changes to the standard streamlit UI/UX
# --> optional, not relevant for the functionality of the component!
#st.set_page_config(page_title="streamlit_audio_recorder")
# Design move app further up and remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
            unsafe_allow_html=True)
# Design change st.Audio to fixed height of 45 pixels
st.markdown('''<style>.stAudio {height: 45px;}</style>''',
            unsafe_allow_html=True)
# Design change hyperlink href link color
st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
            unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
            unsafe_allow_html=True)  # lightmode


# def audiorec_demo_app():

# TITLE and Creator information
st.title('streamlit audio recorder')
st.markdown('Implemented by '
    '[Stefan Rummer](https://www.linkedin.com/in/stefanrmmr/) - '
    'view project source code on '
            
    '[GitHub](https://github.com/stefanrmmr/streamlit-audio-recorder)')
st.write('\n\n')

# TUTORIAL: How to use STREAMLIT AUDIO RECORDER?
# by calling this function an instance of the audio recorder is created
# once a recording is completed, audio data will be saved to wav_audio_data

# wav_audio_data = st_audiorec() # tadaaaa! yes, that's it! :D

# add some spacing and informative messages
col_info, col_space = st.columns([0.57, 0.43])
with col_info:
    st.write('\n')  # add vertical spacer
    st.write('\n')  # add vertical spacer
    st.write('The .wav audio data, as received in the backend Python code,'
                ' will be displayed below this message as soon as it has'
                ' been processed. [This informative message is not part of'
                ' the audio recorder and can be removed easily] 🎈')

if wav_audio_data is not None:
    # display audio data as received on the Python side
    col_playback, col_space = st.columns([0.58,0.42])
    with col_playback:
        st.audio(wav_audio_data, format='audio/wav')
        
if wav_audio_data is not None:
    # オーディオデータをメモリ上のバイトストリームとして保存
    audio_bytes_io = io.BytesIO(wav_audio_data)
    
    # ファイルとして書き出す場合
    with open('recording.wav', 'wb') as out_file:
        out_file.write(audio_bytes_io.getvalue())


# if __name__ == '__main__':
#     # call main function
#     audiorec_demo_app()


# def transcribe(audio_file):
#     transcript = openai.Audio.transcribe("whisper-1", audio_file)
#     return transcript


# def summarize(text):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=(
#             f"Please do what you are asked to do with the following text:\n"
#             f"{text}"
#         ),
#         temperature=0.5,
#         max_tokens=560,
#     )

#     return response.choices[0].text.strip()

# st.image("https://asesorialinguistica.online/wp-content/uploads/2023/04/Secretary-GPT.png")

# st.text("Click on the microphone and tell your GPT secretary what to type.")


# st.sidebar.title("Secretary GPT")

# # Explanation of the app
# st.sidebar.markdown("""
# ## 使用方法
# 0. このサイトがマイクにアクセスできるように、ブラウザの設定を確認してください。
# 1. 音声を録音するか、オーディオファイルをアップロードするかを選択してください。
# 2. 録音を開始する前に、に記録してほしい内容を指示してください：メール、報告書、記事、エッセイなど。
# 3. 音声を録音する場合は、マイクのアイコンをクリックして開始し、もう一度クリックして終了してください。
# 4. オーディオファイルをアップロードする場合は、デバイスからファイルを選択してアップロードしてください。
# 5. 「Transcribe（書き起こし）」をクリックします。待機時間は録音時間に比例します。
# 6. 最初に書き起こしたテキストが表示され、その後にリクエストが続きます。
# 7. 生成されたドキュメントをテキスト形式でダウンロードしてください。
# -  By Moris Polanco
#         """)

# # tab record audio and upload audio
# tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

# with tab1:
#     audio_bytes = audio_recorder(pause_threshold=180)
#     if audio_bytes:
#         st.audio(audio_bytes, format="audio/wav")
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#         # save audio file to mp3
#         with open(f"audio_{timestamp}.mp3", "wb") as f:
#             f.write(audio_bytes)

# with tab2:
#     audio_file = st.file_uploader("Upload Audio", type=["mp3", "mp4", "wav", "m4a"])

#     if audio_file:
#         # st.audio(audio_file.read(), format={audio_file.type})
#         timestamp = timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         # save audio file with correct extension
#         with open(f"audio_{timestamp}.{audio_file.type.split('/')[1]}", "wb") as f:
#             f.write(audio_file.read())

# if st.button("Transcribe"):
#     # check if audio file exists
#     if not any(f.startswith("audio") for f in os.listdir(".")):
#         st.warning("Please record or upload an audio file first.")
#     else:
#         # find newest audio file
#         audio_file_path = max(
#             [f for f in os.listdir(".") if f.startswith("audio")],
#             key=os.path.getctime,
#     )
        

#     # transcribe
#     audio_file = open(audio_file_path, "rb")

#     transcript = transcribe(audio_file)
#     text = transcript["text"]

#     st.subheader("Transcript")
#     st.write(text)

#     # summarize
#     summary = summarize(text)

#     st.subheader("Document")
#     st.write(summary)

#     # save transcript and summary to text files
#     with open("transcript.txt", "w") as f:
#         f.write(text)

#     with open("summary.txt", "w") as f:
#         f.write(summary)

#     # download transcript and summary
#     st.download_button('Download Document', summary)

# # delete audio and text files when leaving app
# if not st.session_state.get('cleaned_up'):
#     files = [f for f in os.listdir(".") if f.startswith("audio") or f.endswith(".txt")]
#     for file in files:
#         os.remove(file)
#     st.session_state['cleaned_up'] = True