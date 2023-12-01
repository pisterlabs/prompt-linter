# langchain: https://python.langchain.com/
from dataclasses import dataclass
import streamlit as st
from speech_recognition.openai_whisper import save_wav_file, transcribe
from audio_recorder_streamlit import audio_recorder
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationChain
from langchain.prompts.prompt import PromptTemplate
from prompts.prompts import templates
from typing import Literal
#from aws.synthesize_speech import synthesize_speech
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from PyPDF2 import PdfReader
from prompts.prompt_selector import prompt_sector
from streamlit_lottie import st_lottie
import json
from IPython.display import Audio
import nltk
from PIL import Image
### ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# im = Image.open("icc.png")
# st.image(im,width=50)
st.markdown("#### Bienvenue, nous allons discuter de vos expériences professionelles")
st.markdown("""\n""")
#position = st.selectbox("Sélectionnez le poste pour lequel vous postulez", ["Data Analyst", "Ingénieur Logiciel", "Marketing"])
position="Data Analyst"
resume = st.file_uploader("Téléchargez votre CV", type=["pdf"])
if 'name_surname' in st.session_state:
    st.write(f"Nom: {st.session_state.name_surname}")
else:
    st.write("Merci de préciser le nom à l'accueil")
#auto_play = st.checkbox("Let AI interviewer speak! (Please don't switch during the interview)")

#st.toast("4097 tokens is roughly equivalent to around 800 to 1000 words or 3 minutes of speech. Please keep your answer within this limit.")
### ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
@dataclass
class Message:
    """Class for keeping track of interview history."""
    origin: Literal["human", "ai"]
    message: str

with st.sidebar:
    st.markdown("IDIAP Create Challenge 2023")
def save_vector(resume):
    """embeddings"""
    nltk.download('punkt')
    pdf_reader = PdfReader(resume)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # Split the document into chunks
    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)
    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch

def initialize_session_state_resume():
    # convert resume to embeddings
    if 'docsearch' not in st.session_state:
        st.session_state.docserch = save_vector(resume)
    # retriever for resume screen
    if 'retriever' not in st.session_state:
        st.session_state.retriever = st.session_state.docserch.as_retriever(search_type="similarity")
    # prompt for retrieving information
    if 'chain_type_kwargs' not in st.session_state:
        st.session_state.chain_type_kwargs = prompt_sector(position, templates)
    # interview history
    if "resume_history" not in st.session_state:
        st.session_state.resume_history = []
        st.session_state.resume_history.append(Message(origin="ai", message="Bonjour, je suis votre intervieweur aujourd'hui. Je vais vous poser quelques questions concernant votre CV et votre expérience. Êtes-vous prêt à commencer l'entretien?"))
    # token count
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    # memory buffer for resume screen
    if "resume_memory" not in st.session_state:
        st.session_state.resume_memory = ConversationBufferMemory(human_prefix = "Candidate: ", ai_prefix = "Interviewer")
    # guideline for resume screen
    if "resume_guideline" not in st.session_state:
        llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.5,)

        st.session_state.resume_guideline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type_kwargs=st.session_state.chain_type_kwargs, chain_type='stuff',
            retriever=st.session_state.retriever, memory = st.session_state.resume_memory).run("Vous êtes l'intervieweur pour un poste d'analyste de données. Ceci est le CV d'un candidat. Posez-lui une seule question technique liée à son expérience et au poste d'analyste de données. Posez-lui une seule question.")
    # llm chain for resume screen
    if "resume_screen" not in st.session_state:
        llm = ChatOpenAI(
            #model_name="gpt-3.5-turbo",
            temperature=0.7, )

        PROMPT = PromptTemplate(
            input_variables=["history", "input"],
            template= """Je veux que vous agissiez comme un intervieweur en suivant strictement la directive dans la conversation actuelle.

                        Posez-moi une questions et attendez ma réponse comme le ferait un humain. Ne rédigez pas d'explications.
                        Le candidat n'a pas accès à la directive.
                        Ne posez qu'une seule question à la fois.
                        N'hésitez pas à poser des questions de suivi si vous le jugez nécessaire.
                        Ne posez pas la même question.
                        Ne répétez pas la question.
                        Le candidat n'a pas accès à la directive.
                        Votre nom est GPTInterviewer.
                        Je veux que vous ne répondiez que comme un intervieweur.
                        Ne rédigez pas toute la conversation en une fois.
                        Le candidat n'a pas accès à la directive.
                        
                        Conversation actuelle :
                        {history}
                        
                        Candidat : {input}
                        IA : """)
        st.session_state.resume_screen =  ConversationChain(prompt=PROMPT, llm = llm, memory = st.session_state.resume_memory)
    # llm chain for generating feedback
    if "resume_feedback" not in st.session_state:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5,)
        st.session_state.resume_feedback = ConversationChain(
            prompt=PromptTemplate(input_variables=["history","input"], template=templates.feedback_template),
            llm=llm,
            memory=st.session_state.resume_memory,
        )

def answer_call_back():
    with get_openai_callback() as cb:
        # user input
        human_answer = st.session_state.answer
        # transcrire audio
        if voice:
            save_wav_file("temp/audio.wav", human_answer)
            try:
                input = transcribe("temp/audio.wav")
            except:
                st.session_state.resume_history.append(Message("ai", "Désolé, je n'ai pas compris."))
                return "Veuillez réessayer."
        else:
            input = human_answer

        st.session_state.resume_history.append(
            Message("human", input)
        )
        # OpenAI answer and save to history
        llm_answer = st.session_state.resume_screen.run(input)
        # speech synthesis and speak out
        #audio_file_path = synthesize_speech(llm_answer)
        # create audio widget with autoplay
        #audio_widget = Audio(audio_file_path, autoplay=True)
        # save audio data to history
        st.session_state.resume_history.append(
            Message("ai", llm_answer)
        )
        st.session_state.token_count += cb.total_tokens
        #return audio_widget

### ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

# sumitted job description
if position and resume:
    # initialise l'état de la session
    initialize_session_state_resume()
    credit_card_placeholder = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        feedback = st.button("Obtenir un retour sur l'entretien")
    with col2:
        guideline = st.button("Montrez-moi le guide d'entretien!")
    chat_placeholder = st.container()
    answer_placeholder = st.container()
    audio = None
    # si soumet une adresse e-mail, obtenir un retour sur l'entretien immédiatement
    if guideline:
        st.markdown(st.session_state.resume_guideline)
    if feedback:
        evaluation = st.session_state.resume_feedback.run("veuillez donner une évaluation concernant l'entretien")
        st.markdown(evaluation)
        st.download_button(label="Télécharger le retour sur l'entretien", data=evaluation, file_name="retour_entretien.txt")
        st.stop()
    else:
        with answer_placeholder:
            voice: bool = st.checkbox("Je souhaite parler à l'intervieweur IA!")
            if voice:
                answer = audio_recorder(pause_threshold=2, sample_rate=44100)
            else:
                answer = st.chat_input("Votre réponse")
            if answer:
                st.session_state['answer'] = answer
                audio = answer_call_back()

        with chat_placeholder:
            for answer in st.session_state.resume_history:
                if answer.origin == 'ai':
                    #if auto_play and audio:
                     #   with st.chat_message("assistant"):
                      #      st.write(answer.message)
                      #      st.write(audio)
                    #else:
                    with st.chat_message("assistant"):
                        st.write(answer.message)
                else:
                    with st.chat_message("user"):
                        st.write(answer.message)

        credit_card_placeholder.caption(f"""
                        Progress: {int(len(st.session_state.resume_history) / 30 * 100)}% completed.""")

