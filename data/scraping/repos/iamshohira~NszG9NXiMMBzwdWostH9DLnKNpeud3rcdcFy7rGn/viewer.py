import streamlit as st
import json
import openai, os
from cryptography.fernet import Fernet
# from st_audiorec import st_audiorec
from audio_recorder_streamlit import audio_recorder
from tempfile import NamedTemporaryFile
openai.api_key = os.environ["OPENAI_API_KEY"]

st.set_page_config(layout="wide")

def transcribe_audio_to_text(audio_bytes):
    with NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        temp_file.flush()
        with open(temp_file.name, "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file, language='en')
    return response.text

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == os.environ["ST_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True
    
def grammer_check(text):
    prompt = "あなたは英語の教師です。ユーザーの発話が英文法的に正しいかを確認し、正しくない場合は正しい英語に直して下さい。"
    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=msg)
    return response.choices[0].message.content

def translator(text):
    prompt = "あなたは英語の教師です。ユーザーが英語話で困っているので英単語や例文を教えてください。"
    msg = [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=msg)
    return response.choices[0].message.content

def reset_message(type="Discussion", question=""):
    if type == "Discussion":
        prompt = f'''You are an English teather. Let's discuss with your student:
                This is a speaking training for the student. So you should be a listener and your statement must be short, less than 50 words.
                '''
        return [{"role": "system", "content": prompt}, {"role": "assistant", "content": question}]
    elif type == "Role Play":
        prompt = f'''You are an English teather. Let's role play with your student. 
                This is a speaking training for the student. So you should be a listener and your statement must be short, less than 50 words.
                The following is the situation shown to the student:
                {question}'''
        return [{"role": "system", "content": prompt}]

if "data" not in st.session_state:
    key = os.environ["CRYPT_KEY"].encode()
    f = Fernet(key)
    with open("encrypted_intermediate_with_story", "rb") as fp:
        encrypted_data = fp.read()
    decrypted_data = f.decrypt(encrypted_data)
    st.session_state["data"] = json.loads(decrypted_data.decode())
    st.session_state["lessons"] = [f"Lessson {i+1}: {v['title']}" for i, v in enumerate(st.session_state.data)]

if "messagesD" not in st.session_state:
    st.session_state["messagesD"] = reset_message()

if "messagesR" not in st.session_state:
    st.session_state["messagesR"] = reset_message()

if "selected_theme" not in st.session_state:
    st.session_state["selected_theme"] = None

if "selected_situation" not in st.session_state:
    st.session_state["selected_situation"] = None

if "is_recording" not in st.session_state:
    st.session_state["is_recording"] = False

if "reviews" not in st.session_state:
    st.session_state["reviews"] = {}

if "audio" not in st.session_state:
    st.session_state["audio"] = None


st.title("📖")

if check_password():
    with st.sidebar:
        # translator
        with st.form("translator", clear_on_submit=True):
            st.subheader("Translator")
            res = st.text_area("Input")
            btn = st.form_submit_button("Translate")
        if btn:
            st.write(translator(res))        
    # dropdown list for selecting the lesson
    lesson = st.selectbox("Lesson", st.session_state.lessons)
    if lesson:
        lesson_index = st.session_state.lessons.index(lesson)
        st.session_state["selected_lesson"] = st.session_state.data[lesson_index]
        exs = st.session_state.selected_lesson["exercises"]
        tabs = st.tabs([ex['type'] for ex in exs])
        for ex, tab in zip(exs, tabs):
            with tab:
                if ex['type'] == "Vocabulary":
                    for i, item in enumerate(ex["items"]):
                        with st.expander(item["definition"]):
                            if st.checkbox(item["word_ja"]):
                                st.write(item["word_en"])
                            for j, ex in enumerate(item["examples"]):
                                if st.checkbox(ex["ja"], key=f'example{i}:{j}'):
                                    st.write(ex["en"])
                elif ex['type'] == "Useful Expressions":
                    for i, item in enumerate(ex["items"]):
                        if st.checkbox(item["ja"], key=f'expression{i}'):
                            st.write(item["en"])
                elif ex['type'] == "Dialogue":
                    st.info(ex["situation"]["en"])
                    st.subheader("Character")
                    role = {ex["characters"][0]: "assistant", ex["characters"][1]: "user"}
                    for k, v in role.items():
                        st.chat_message(v).write(k)
                    st.subheader("Dialogue")
                    for i, d in enumerate(ex["dialogue"]):
                        with st.chat_message(role[d["character"]]):
                            if st.checkbox(d["serif"]["ja"], key=f'serif{i}'):
                                st.write(d["serif"]["en"])
                    st.subheader("Question")
                    for i, q in enumerate(ex["questions"]):
                        st.write(q["en"])
                elif ex['type'] == "Listening Comprehension":
                    with open(os.path.join("audio", f"ex{lesson_index}.wav"), "rb") as fp:
                        audio = fp.read()
                    st.audio(audio, format="audio/wav")
                    story = ex["story"]
                    for i, item in enumerate(ex["items"]):
                        st.radio(f"Q{i+1}: {item['question']['en']}", ["A: "+item["answers"][0]["en"], "B: "+item["answers"][1]["en"]])
                    with st.expander("Answer"):
                        st.write(story)
                        st.write(" ".join(["A", "B"][i] for i in ex["answers"]))
                elif ex['type'] in ["Discussion","Role Play"]:
                    cols = st.columns(2)
                    with cols[1]:
                        if ex['type'] == "Discussion":
                            themes = [i['en'] for i in ex["items"]]
                            theme = st.selectbox("Theme", themes)
                            if theme != st.session_state.selected_theme:
                                st.session_state["messagesD"] = reset_message(ex['type'], theme)
                                st.session_state["selected_theme"] = theme
                        elif ex['type'] == "Role Play":
                            situation = "\n".join(["- "+i for i in ex["situation"]["en"]])
                            st.info(situation)
                            if situation != st.session_state.selected_situation:
                                st.session_state["selected_situation"] = situation
                                st.session_state["messagesR"] = reset_message(ex['type'], situation)
                        container = st.empty()
                        # if st.session_state.is_recording:
                        #     if container.button('Stop', type='primary', key=f"stop {ex['type']}"):
                        #         st.session_state["is_recording"] = False
                        #         container.empty()
                        #         container.button('Record', key=f"start {ex['type']}")
                        #         st.session_state['prompt'] = "test"
                        # else:
                        #     if container.button('Record', key=f"start {ex['type']}"):
                        #         st.session_state["is_recording"] = True
                        #         st.info("Recording...")
                        #         container.empty()
                        #         container.button('Stop', type='primary', key=f"stop {ex['type']}")
                        # audio = audiorecorder("Voice input", "Recording...")
                        # audiodata = audio.export().read()
                        # if audiodata != st.session_state.audio:
                        #     st.session_state.audio = audiodata
                        #     with open("audio.mp3", "wb") as fp:
                        #         fp.write(audiodata)
                        #     with open("audio.mp3", "rb") as fp:
                        #         transcript = openai.Audio.transcribe("whisper-1", fp, language="en")
                        #     st.session_state['prompt'] = transcript.text
                        wav_audio_data = audio_recorder(pause_threshold=30,key=f"mic {ex['type']}")
                        if wav_audio_data:
                            if wav_audio_data != st.session_state.audio:
                                st.session_state.audio = wav_audio_data
                                # with open("audio.wav", "wb") as fp:
                                #     fp.write(wav_audio_data)
                                # with open("audio.wav", "rb") as fp:
                                #     transcript = openai.Audio.transcribe("whisper-1", fp, language="en")
                                transcript = transcribe_audio_to_text(wav_audio_data)
                                st.session_state[f"prompt {ex['type']}"] = transcript
                        with st.form(f"input {ex['type']}", clear_on_submit=False):
                            st.text_area("Your response", key=f"prompt {ex['type']}")
                            gc = st.checkbox("Grammer check", value=True, key=f"gc {ex['type']}")
                            submitted = st.form_submit_button("Send")
                    with cols[0]:
                        for msg in st.session_state[f"messages{ex['type'][0]}"]:
                            if msg['role'] == "system":
                                continue
                            with st.chat_message(msg["role"]):
                                prompt = msg["content"]
                                st.write(prompt)
                                if prompt in st.session_state.reviews:
                                    st.info(st.session_state.reviews[prompt])
                        if submitted:
                            prompt = st.session_state[f"prompt {ex['type']}"]
                            msg = {"role": "user", "content": prompt}
                            st.session_state[f"messages{ex['type'][0]}"].append(msg)
                            with st.chat_message(msg["role"]):
                                prompt = msg["content"]
                                st.write(prompt)
                                if gc:
                                    review = grammer_check(prompt)
                                    st.session_state.reviews[prompt] = review
                                    st.info(review)
                            ai_response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state[f"messages{ex['type'][0]}"])
                            msg = ai_response.choices[0].message
                            st.session_state[f"messages{ex['type'][0]}"].append(msg)
                            st.chat_message("assistant").write(msg.content)
                    css='''
                        <style>
                        div[data-testid="column"] {
                            display: flex; 
                            flex-direction: column-reverse;
                            overflow-y: auto;
                            height: 70vh;
                        }
                        </style>
                        '''
                    st.markdown(css, unsafe_allow_html=True)

