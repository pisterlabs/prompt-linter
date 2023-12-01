import streamlit as st
import openai
import requests
import base64
import json
from audio_recorder_streamlit import audio_recorder
# import boto3
import time

import streamlit_authenticator as stauth

from st_supabase_connection import SupabaseConnection

# from supabase import create_client, Client

import yaml
from yaml.loader import SafeLoader

st.set_page_config(page_title="EH Speaking Homework", page_icon="🐧")


# Initialize connection.
# Uses st.cache_resource to only run once.

# @st.cache_resource
# def init_connection():
#     url = st.secrets["supabase_url"]
#     key = st.secrets["supabase_key"]
#     return create_client(url, key)


# supabase = init_connection()

st_supabase_client = st.experimental_connection(
    name="supabase_connection",
    type=SupabaseConnection,
    ttl=None,
    url=st.secrets["supabase_url"],
    key=st.secrets["supabase_key"],
)

# CONNECT TO AWS

# your AWS keys
# aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
# aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]

# initialize s3 client, be sure aws_access_key_id & aws_secret_access_key are environment variables

# s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

# --- USER AUTHENTICATION ---

with open('config.yaml', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

if "username" not in st.session_state:
    st.session_state["username"] = ""

name, authentication_status, username = authenticator.login("Login", "main")

if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None  # or your default value

if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'sidebar', key='unique_key')

    username = st.session_state["username"]

    openai.api_key = st.secrets["OPENAI_API_KEY"]
    azure_key = st.secrets["AZURE_SUBSCRIPTION_KEY"]
    region = st.secrets["SPEECH_REGION"]

    st.title("Speaking Homework")
    st.sidebar.markdown(
        """
        ### 💡 Instructions 
        1. Select your level
        2. Select your class
        3. Select a lesson
        4. Select a question.
        5. Hit the Record button to record. Hit it again to stop.
        6. Press **Submit** to to receive pronunciation score and AI feedback.
        """
    )
    st.sidebar.info(
        "You must answer all questions in a lesson unless instructed otherwise.")

    # AZURE TEXT TO SPEECH (https://github.com/hipnologo/openai_azure_text2speech)

    def get_azure_access_token():
        try:
            response = requests.post(
                "https://%s.api.cognitive.microsoft.com/sts/v1.0/issuetoken" % region,
                headers={
                    "Ocp-Apim-Subscription-Key": azure_key
                }
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
            return None

        return response.text

    def text_to_speech(text, voice_name='en-GB-RyanNeural'):
        access_token = get_azure_access_token()

        if not access_token:
            return None

        try:
            response = requests.post(
                "https://%s.tts.speech.microsoft.com/cognitiveservices/v1" % region,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/ssml+xml",
                    "X-MICROSOFT-OutputFormat": "riff-24khz-16bit-mono-pcm",
                    "User-Agent": "TextToSpeechApp",
                },
                data=f"""
                    <speak version='1.0' xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang='en-US'>
                    <voice name='{voice_name}'>
                    <mstts:express-as role="YoungAdultFemale" style="cheerful">
                        {text}
                    </mstts:express-as>
                    </voice>
                    </speak>
                """,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
            return None

        return response.content

    st.write(f"#### Welcome, {name}! 🥳")

    user_level = st.radio(
        "Select your level",
        ('Advanced', 'Pre-Intermediate',))

    if user_level == 'Pre-Intermediate':
        user_class = st.radio(
            "Select your class",
            ('PI.074',))
    elif user_level == 'Advanced':
        user_class = st.radio(
            "Select your class",
            ('A.058',))

    lesson_number = st.selectbox(
        'Select a lesson',
        ('Lesson 1', 'Lesson 2', 'Lesson 3', 'Lesson 4', 'Lesson 5', 'Lesson 6', 'Lesson 7', 'Lesson 8', 'Lesson 9', 'Lesson 10', 'Lesson 11', 'Lesson 12', 'Midterm 1', 'Midterm 2', 'Midterm 3', 'Midterm 4', 'Midterm 5', 'Midterm 6', 'Midterm 7', 'Midterm 8'))

    # Define a dictionary to store the questions for each level and lesson
    question_bank = {
        'Advanced': {
            'Lesson 1': ['Tell me about the kind of accommodation you live in?', 'How long have you lived there?', 'What do you like about living there?', 'What sort of accommodation would you most like to live in?'],
            'Lesson 2': ['Do you like science?', 'When did you start to learn about science?', 'Which science subject is interesting to you?', 'What kinds of interesting things have you done with science?'],
            'Lesson 3': ['What study tools do you use?', 'What parts of your hometown do you recommend a visitor go and see?', 'What kind of decorations does your room have?', 'What kind of work are you planning to do in the future?'],
            'Lesson 4': ['Describe a popular music group (or a singer) in your country.\n\nYou should say: \n\nwhat it is\n\nwho the lead singer is\n\nwhen their debut was\n\nand explain why they are famous in your country.',],
            'Lesson 5': ['Describe a time when you tried to do something but was not very successful.\n\nYou should say:\n\nwhen it was\n\nwhat you tried\n\nwhy it was not very successful\n\nand how you felt about it.',],
        },
        'Pre-Intermediate': {
            'Lesson 1': ['What work would you like to do after you finish your studies?', 'What do you usually do in your room?', 'What place (city) would you like to live in in the future?'],
            'Lesson 2': ['Do you like to plan what you will do each day?', 'What\'s usually your busiest time of the day?', 'What kind of weather do you dislike?'],
            'Lesson 3': ['Do you ever receive gifts?', 'What sorts of gifts do you give to your family and friends?', 'When you were a child, what places did you go to for entertainment?'],
            'Lesson 4': ['Do you like spending your leisure time alone?', 'Do you think we need to take a nap in the afternoon?', 'Do you do anything before going to bed to help you sleep?'],
            'Lesson 5': ['Describe something you do to stay healthy.\n\nYou should say:\n\nwhat you do\n\nwhen you started doing this\n\nhow much time you spend doing this\n\nand explain how this activity helps you stay healthy.',],
            'Lesson 6': ['Describe a place that you visited for a short time and would like to revisit. You should say:\n\nwhere / what this place was\n\nhow long you spent there\n\nwhat you did there\n\nand explain why you would like to go back there again.',],
            'Lesson 7': ['Describe a person who is helpful to you with your studies or at work. You should say:\n\nwho this person is\n\nwhere they help you\n\nhow this person helps you\n\nand explain how you feel about this help (or this person).',],
            'Lesson 8': ['Describe an interesting part of your country. You should say:\n\nwhere it is\n\nhow you know about it\n\nwhether or not you have visited this place\n\nand explain why you think it is interesting.',],
            'Lesson 9': ['Do you think you are a good public speaker?', 'Can you suggest why some children like talking while others don\'t like to talk so much?',],
            'Lesson 10': ['Some people say young people spend too much time on shopping. What do you think?', 'Do you think the way we judge success has changed, compared to the way it was judged in the past?',],
            'Lesson 11': ['Do you think it\'s a good thing for people to live in high-rise apartment buildings?',],
            'Lesson 12': ['What types of news are you most interested in?', 'Do you think the way we get our news might change in the future?',],
            'Midterm 1': ['Describe a person who you would like to be similar to in the future.\n\nYou should say:\n\nwho this person is\n\nhow you know this person\n\nwhat impresses you about this person\n\nand explain why you would like to be similar to this person.',],
            'Midterm 2': ['Describe a leisure activity you enjoy doing with your family.\n\nYou should say:\n\nwhat activity\n\nwhen you do it\n\nwhere you do it\n\nand explain why you enjoy it.',],
            'Midterm 3': ['Describe a memorable holiday you had.\n\nYou should say:\n\nwhen and where you went\n\nwho you went with\n\nwhat you did there\n\nand explain why it was memorable.',],
            'Midterm 4': ['Describe a useful practical skill that you learned (such as driving a car, speaking a foreign language, cooking etc).\n\nYou should say:\n\nwhy you learned this skill\n\nhow (and when) you learned it\n\nhow difficult it was to learn\n\nand explain how this skill is useful to you.',],
            'Midterm 5': ['Describe your favourite weather.\n\nYou should say:\n\nwhat kind of weather it is \n\nwhen this weather usually occurs\n\nwhat you usually do during this weather \n\nand explain why you like this type of weather.',],
            'Midterm 6': ['Describe your favourite part of your hometown, (or where you are living now).\n\nYou should say:\n\nwhere it is\n\nhow often you go there\n\nwhat people do there\n\nand explain why you like it so much.',],
            'Midterm 7': ['Describe a place where you would like to have (or, build) a home.\n\nYou should say:\n\nwhere it is\n\nwhat the place looks like\n\nwhat work you would do if you lived there\n\nand explain why you would like to live there.',],
            'Midterm 8': ['Describe a recent event that made you feel happy.\n\nYou should say:\n\nwhat the event was\n\nwhere it happened.\n\nwho was with you\n\nand explain why this event was so enjoyable.'],
        },
        # Add other levels as needed
    }

    # Get the selected questions based on the level and lesson
    selected_questions = question_bank[user_level][lesson_number]

    # Generate question number labels for each question
    if len(selected_questions) == 1:
        question_number_labels = ['Question']
    else:
        question_number_labels = ['Question {}'.format(
            i+1) for i in range(len(selected_questions))]

    selected_question_number = st.selectbox(
        'Select a question', question_number_labels)

    # Display the selected question with st.write
    selected_question_index = question_number_labels.index(
        selected_question_number)
    question = selected_questions[selected_question_index]
    st.info(question)

    # AUDIO RECORDER

    audio_bytes = audio_recorder(
        energy_threshold=(-1.0, 1.0),
        pause_threshold=120.0,
        sample_rate=32_000,
        text="",
        recording_color="#009900",
        neutral_color="#777777",
        icon_name="microphone",
        icon_size="3x",
    )
    st.caption("👆 Click to record. Click again to stop.")

    user_answer = ""

    if audio_bytes:
        file_name = "temp_audio_file.wav"

        with open(file_name, mode="wb") as recorded_data:
            recorded_data.write(audio_bytes)
            st.audio(audio_bytes, format="audio/wav")

        # generating unique file name using timestamp

        # timestamp = str(time.time())
        # unique_file_name = "audio_" + username + "_" + timestamp + ".wav"

        # upload file to s3
        # bucket_name = 'lingocopilot'
        # s3.upload_file(file_name, bucket_name, unique_file_name)

        if st.button("Submit"):
            with st.spinner('🦻 Transcribing and assessing pronunciation...'):
                if audio_bytes:
                    # generating unique file name using timestamp
                    timestamp = str(time.time())
                    formatted_time = time.strftime(
                        "%H%M%S", time.localtime(float(timestamp)))
                    unique_file_name = "audio_" + username + "_" + formatted_time + ".wav"

                    file_name = unique_file_name  # Use the unique filename directly

                    with open(file_name, mode="wb") as recorded_data:
                        recorded_data.write(audio_bytes)

                    supabase_destination = f"{user_class}/{lesson_number}/" + \
                        unique_file_name

                    st_supabase_client.upload(
                        "st.lingocopilot", source="hosted", file=file_name, destination_path=supabase_destination)

                    audio_url = st_supabase_client.get_public_url(
                        "st.lingocopilot", filepath=supabase_destination)

                    # Clear the local file after successful upload
                    # os.remove(file_name)
                with open("temp_audio_file.wav", "rb") as wav_file:
                    transcript = openai.Audio.transcribe(
                        "whisper-1", wav_file, prompt="Transcribe this into English.")
                user_answer = transcript.text
                # st.markdown(f"💬 *{user_answer}*")
                # st.write('\n\n')

                def get_chunk(audio_source, chunk_size=1024):
                    # yield WaveHeader16K16BitStereo
                    while True:
                        # time.sleep(chunk_size / 32000)
                        chunk = audio_source.read(chunk_size)
                        if not chunk:
                            # global uploadFinishTime
                            # uploadFinishTime = time.time()
                            break
                        yield chunk

                referenceText = user_answer
                pronAssessmentParamsJson = "{\"ReferenceText\":\"%s\",\"GradingSystem\":\"HundredMark\",\"Dimension\":\"Comprehensive\",\"PhonemeAlphabet\":\"IPA\"}" % referenceText
                pronAssessmentParamsBase64 = base64.b64encode(
                    bytes(pronAssessmentParamsJson, 'utf-8'))
                pronAssessmentParams = str(pronAssessmentParamsBase64, "utf-8")

                url = "https://%s.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-us" % region
                headers = {'Accept': 'application/json;text/xml',
                           'Connection': 'Keep-Alive',
                           'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
                           'Ocp-Apim-Subscription-Key': azure_key,
                           'Pronunciation-Assessment': pronAssessmentParams,
                           'Transfer-Encoding': 'chunked',
                           'Expect': '100-continue'}

                with open("temp_audio_file.wav", "rb") as wav_file:
                    response = requests.post(
                        url=url, data=get_chunk(wav_file), headers=headers)

                # getResponseTime = time.time()

                resultJson = json.loads(response.text)
                # print(json.dumps(resultJson, indent=4))

                accuracy_score = resultJson["NBest"][0]["AccuracyScore"]
                fluency_score = resultJson["NBest"][0]["FluencyScore"]
                pron_score = resultJson["NBest"][0]["PronScore"]
                mis_pron_words = [
                    word["Word"]
                    for word in resultJson["NBest"][0]["Words"]
                    if word["ErrorType"] == "Mispronunciation"
                ]
                col1, col2, col3 = st.columns([1, 1, 2])
                # col1.metric("Accuracy", accuracy_score)
                col1.metric("Fluency", fluency_score)
                col2.metric("Pronunciation", pron_score)
                mispronunciation = ""
                if mis_pron_words:
                    mispronunciation = ", ".join(mis_pron_words)
                    col3.write(f"**MISPRONUNCIATION:** {mispronunciation}")
            with st.spinner('💬 Improving answer...'):
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": """You are a native English speaker. You use spoken English in an everyday conversational tone. Your vocabulary is that of a high school student. Your must not use written English such as "furthermore", "therefore", "additionally" "overall", and "in conclusion"."""},
                        {"role": "user", "content": f"""Improve the following IELTS Speaking answer by a candidate with simple, natural English. Do not add new ideas. Just give me the improved answer. No commentary. Here is the question: {question}. And here's the candidate's answer: {user_answer}"""}
                    ]
                )
            improved_answer = response.choices[0].message.content.strip()
            st.markdown("### Improved answer")
            st.success(f"✨ {improved_answer}")
            with st.spinner('🔈 Generating audio and extracting expression...'):
                improved_answer_audio = text_to_speech(
                    improved_answer, 'en-GB-RyanNeural')
                with open('improved_answer_audio.wav', 'wb') as f:
                    f.write(improved_answer_audio)
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "user", "content": f"""Extract one good collocation from the an IELTS Speaking part 1 answer, accompanied by a definition and an easy and clear example. Use this format:
                        - **[expression]**: [definition]
                        - *[example]*
                        Here is the answer: {improved_answer}"""}
                    ]
                )
            idiomatic_exp = response.choices[0].message.content.strip()
            st.audio('improved_answer_audio.wav')
            st.markdown("### Collocation")
            st.markdown(idiomatic_exp)
            # with st.spinner('✍️ Generating detailed feedback...'):
            #     response = openai.ChatCompletion.create(
            #         model = 'gpt-4',
            #         messages = [
            #             {"role": "system", "content": """You are an English teacher."""},
            #             {"role": "user", "content": f"""Compare the following answer by a learner with its improved version as if you were speaking to a 9-year-old. You must point out in detail what aspects of the original answer have been improved, along with relevant excerpts from the improved answer. Here's the original answer: "{user_answer}". Here's the improved answer: "{improved_answer}". And here's the question for context: "{question}". Give me a markdown table with two columns: What was improved and Excerpt. """}
            #         ]
            #     )
            # detailed_feedback = response.choices[0].message.content.strip()
            # st.markdown("### Feedback")
            # st.success(detailed_feedback)
            st_supabase_client.table("users").upsert(
                {"username": username}).execute()
            st_supabase_client.table("eh_speaking_hw").insert({"username": username, "name": name, "question": question, "user_answer": user_answer, "improved_answer": improved_answer,
                                                               "idiomatic_exp": idiomatic_exp, "accuracy_score": accuracy_score, "fluency_score": fluency_score, "pron_score": pron_score, "mispronunciation": mispronunciation,
                                                               "user_audio": audio_url,
                                                               "user_class": user_class,
                                                               "lesson_number": lesson_number}).execute()

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
