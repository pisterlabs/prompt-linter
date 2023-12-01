import random
import streamlit as st
import os
from dotenv import load_dotenv
import openai
from audiorecorder import audiorecorder
from streamlit.components.v1 import html

# Import the AssemblyAI module
import assemblyai as aai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Small talk buddy",
    page_icon="🗣️",
)

# add the ideogram.jpeg on the page
st.image("ideogram.jpeg")

phrases = [
    "Hi there! What's the most interesting thing that happened to you today?",
    "What's your favorite way to spend a lazy Sunday?",
    "If you could travel anywhere in the world right now, where would you go?",
    "What's the best book you've read recently, and why did you enjoy it?",
    "Do you have any exciting plans for the weekend?",
    "What's the most memorable meal you've ever had?",
    (
        "If you could have dinner with any historical figure, who would it be"
        " and why?"
    ),
    (
        "What's something you've always wanted to learn but haven't had the"
        " chance to?"
    ),
    "What's your go-to comfort food?",
    "Have you ever had a really strange or unusual job?",
    "Do you believe in aliens or extraterrestrial life?",
    "What's a skill or hobby you wish you had more time for?",
    "If you could switch lives with someone for a day, who would it be?",
    "What's a movie or TV show you could watch over and over again?",
    "Do you have a favorite quote or motto that you live by?",
    "What's the most beautiful place you've ever visited?",
    "If you could change one thing about the world, what would it be?",
    "What's the most adventurous thing you've ever done?",
    "What's your all-time favorite song, and why does it resonate with you?",
    "Are you a morning person or a night owl?",
    "What's the most valuable life lesson you've learned so far?",
    "If you could have any superpower, what would it be and why?",
    "What's your favorite season, and what do you love most about it?",
    "Do you have any upcoming travel plans?",
    "What's a recent accomplishment you're really proud of?",
    "What's your preferred way to relax and de-stress after a long day?",
    (
        "If you could invite three people, living or dead, to a dinner party,"
        " who would they be?"
    ),
    "What's a small thing that always brings a smile to your face?",
    (
        "What's a place you've always wanted to visit but haven't had the"
        " chance to yet?"
    ),
    "What's the most interesting documentary you've watched recently?",
    "Do you have a favorite childhood memory?",
    "What's your take on the concept of fate or destiny?",
    "What's a hobby or interest you have that might surprise people?",
    "If you could master any skill instantly, what would it be?",
    "What's your philosophy on work-life balance?",
    "What's the last song that got stuck in your head?",
    (
        "If you could have a conversation with your future self, what would"
        " you ask?"
    ),
    "What's the best piece of advice you've ever received?",
    "What's your favorite type of cuisine, and do you have a go-to dish?",
    "What's the most daring thing on your bucket list?",
    "Do you enjoy any outdoor activities, like hiking or camping?",
    "What's a technology or scientific advancement that fascinates you?",
    "If you could live in any era of history, when and where would it be?",
    "What's a recent news story or current event that caught your attention?",
    "What's the most unusual food you've ever tried?",
    "Do you have any pet peeves or things that annoy you easily?",
    "What's a hidden talent you have that not many people know about?",
    (
        "If you could give your younger self one piece of advice, what would"
        " it be?"
    ),
    "What's your favorite way to give back to your community or help others?",
    (
        "What's a dream you've had that you'd love to see come true in the"
        " future?"
    ),
]


def generate_feedback(phrase, transcription_text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Small Talk Coach.\n\nYour role is to analyze"
                    " how well a person replies to small talk"
                    " questions.\n\nThe student is not a"
                    " native-speaker.\n\nYour role is to find errors in the"
                    " student phrases, such as wrong verbal time, wrong use of"
                    " plurals, or even wrong use of some words.\n\nSometimes,"
                    " the answer doesn't have any major problems, but the"
                    " person repeats the same expression or it's not clear."
                    " Your role is to review the answer and provide feedback"
                    " to the user so that they can improve their English and"
                    " how they speak well in public.\n\nPlease provide ideas"
                    " on how the person could speak better.\n\nAt the"
                    " beginning, include any general comments related to the"
                    " whole answer.\nAfter that, if you find mistakes, create"
                    " a list of them, and explain why. You can divide the"
                    " mistakes found in topics, such as:\n\n- Grammar: List"
                    " all the mistakes related to grammar.\n- Use of incorrect"
                    " pronouns.\n- Use of incorrect verbs.\n- Repetitive"
                    " terms.\n- Unclear ideas.\n- etc\n\nIf the answer doesn't"
                    " have the topics above you should not mention them. Also,"
                    " feel free to include other topics.\n\nYour feedback will"
                    " be shared directly to the student, so you can speak"
                    " directly to them."
                ),
            },
            {
                "role": "user",
                "content": (
                    'Question:"'
                    + phrase
                    + '"\nStudent Answer: "'
                    + transcription_text
                    + '"'
                ),
            },
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content


if "random" not in st.session_state:
    st.session_state["random"] = random.randint(0, len(phrases) - 1)

phrase = phrases[st.session_state.random]

st.write("Here is a place where you can practice small talk! 🗣️")

st.write("""
    Just read the question below, hit record, answer the question, and click on stop! 🎙️
""")

st.write("""
    The Small Talk buddy will analyze it and provide feedback on how to improve your small talk game! 📈
""")

st.write(
    """- Try to answer that question below as if you were in a conversation with someone.

- PS: If you don't feel comfortable of talking about this topic, you can refresh the page and get another phrase to practice!."""
)

st.markdown("---")

st.header(phrase)

st.markdown("---")

# Audio recording
audio = audiorecorder("Start recording", "Stop recording")

if len(audio) > 0:
    # To play audio in frontend:
    st.audio(audio.tobytes(), format="audio/wav")

    # To save audio to a file (assuming you want a WAV file):
    wav_file = open("audio.wav", "wb")
    wav_file.write(audio.tobytes())
    wav_file.close()

    # Install the assemblyai package by executing the command `pip3 install assemblyai` (macOS) or `pip install assemblyai` (Windows).

    # Your API token is already set here
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

    with st.spinner("Transcribing your audio..."):
        # Create a transcriber object.
        transcriber = aai.Transcriber()

        # If you have a local audio file, you can transcribe it using the code below.
        # Make sure to replace the filename with the path to your local audio file.
        transcription_text = transcriber.transcribe("audio.wav").text

    st.markdown("---")

    st.header("Your answer")
    st.write(transcription_text)

    st.markdown("---")

    st.header("Feedback")
    with st.spinner("Generating your feedback, please wait..."):
        feedback = generate_feedback(phrase, transcription_text)

    st.write(feedback)

footer = """<style>
.footer {
  width: 56vw;
  position: fixed;
  padding: 1.25rem;
  text-align: center;
  font-size: 0.875rem;
  line-height: 1.25rem;
  bottom: 0;
  background-color: rgb(14, 17, 23);
}

.footer a, .footer a:hover {
  text-decoration: none;
}
</style>
<div class="footer">
  <a
    style="color: white"
    href="https://ae.studio?utm_source=smalltalkbuddy.com"
    target="_blank"
  >
    Made with ❤️ by <span style="text-decoration-line: underline;">AE Studio</span>
  </a>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

html(
    """<script src="https://scripts.simpleanalyticscdn.com/latest.js"></script>"""
)
