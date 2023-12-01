from pathlib import Path
from random import randrange
import base64
import os
import json
import random
from pathlib import Path
from typing import List
import streamlit as st
from dotenv import load_dotenv
## import stt
from src.utils.stt import show_voice_input
from src.utils.load_file import load_file
from PyPDF2 import PdfReader
import langchain
from streamlit_chat import message as msgchat
from conversant.prompts import ChatPrompt
import cohere as cohere_
import conversant


langchain.verbose = False
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Cohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from src.utils.htmlTemplates import css, bot_template, user_template
from streamlit_option_menu import option_menu
from src.styles.menu_styles import FOOTER_STYLES, HEADER_STYLES
from src.utils.gen_quest import QuestionGenerator, Rephraser
from src.utils.evaluate import evaluator


# --- PATH SETTINGS ---
current_dir = os.path.join(os.path.dirname(__file__))
css_file = os.path.join(current_dir, "src/styles/.css")
assets_dir = os.path.join("", "assets")
icons_dir = os.path.join(assets_dir, "icons")
img_dir = os.path.join(assets_dir, "img")
tg_svg = os.path.join(icons_dir, "tg.svg")
lk_svg = os.path.join(icons_dir, "lk.svg")
tw_svg = os.path.join(icons_dir, "tw.svg")


# --- GENERAL SETTINGS ---
PAGE_TITLE: str = "Your Study Assistant"
PAGE_ICON: str = "📚"
load_dotenv()

with open("conversant/data/study.json") as f:
    CONFIG = json.load(f)
CLIENT = cohere_.Client(os.getenv('COHERE_API_KEY'))


# Session States
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "input_kind" not in st.session_state:
    st.session_state.input_kind = "Text"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "bot" not in st.session_state:
    st.session_state.bot = conversant.PromptChatbot(
        client=CLIENT,
        prompt=ChatPrompt.from_dict(CONFIG)
    )
if "qtype" not in st.session_state:
    st.session_state.qtype = ""


# --- FUNCTIONS ---
def get_random_img(img_names: list[str]) -> str:
    return random.choice(img_names)

def render_svg(svg: Path) -> str:
    """Renders the given svg string."""
    with open(svg) as file:
        b64 = base64.b64encode(file.read().encode("utf-8")).decode("utf-8")
        return f"<img src='data:image/svg+xml;base64,{b64}'/>"


def get_files_in_dir(path: Path) -> List[str]:
    files = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files.append(file)
    return files

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, overlap=200, chunk_size=1000):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = CohereEmbeddings(model="search-english-beta-2023-04-02")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = Cohere()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def show_text_input() -> None:
    st.text_area(label="Start your conversation your Study Pal", value=st.session_state.user_text, key="user_text")


def get_user_input():
    match st.session_state.input_kind:
        case "Text":
            show_text_input()
        case "Voice [test mode]":
            show_voice_input()
        case _:
            show_text_input()

def clear_chat() -> None:
    st.session_state.generated = []
    st.session_state.past = []
    st.session_state.messages = []
    st.session_state.user_text = ""
    st.session_state.history = []
    st.session_state.bot = conversant.PromptChatbot(
        client=cohere_.Client(os.getenv('COHERE_API_KEY')),
        prompt=ChatPrompt.from_dict(CONFIG)
    )
    st.session_state.conversation = None


def reset_QA():
    st.session_state.questions = []
    st.session_state.options = []
    st.session_state.answers = []
    st.session_state.conversation = None
    st.session_state.quests = []
    st.session_state.ans = []




def show_chat_buttons() -> None:
    b0, b1, b2 = st.columns(3)
    with b0, b1, b2:
        b0.button(label="Ask")
        b1.button(label="Clear", on_click=clear_chat)
        b2.download_button(
            label="Save",
            data="\n".join([str(d) for d in st.session_state.history[1:]]),
            file_name="ai-talks-chat.json",
            mime="application/json",
        )


def handle_user_input(user_question):

    response = st.session_state.conversation(user_question)
    # response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            msgchat(message.content, key=str(i), is_user=True)

        else:
            msgchat(message.content, key=str(i))

def show_conversation():
    for i, message in enumerate(st.session_state.history):
        if i % 2 == 0:
            msgchat(message, key=str(i), is_user=True)

        else:
            msgchat(message, key=str(i))

def show_info(icon: Path) -> None:
    st.divider()
    st.markdown("<div style='text-align: justify;'>Welcome to Study Assistant! Our platform offers a range of tools to help you learn more effectively. With our chat feature, you can ask questions from your school materials uploaded, then you can also generate questions from documents to aid learning. We also offer text summarization and correction tools to help you save time and improve your writing skills. And if you’re struggling with studying, our studypal is always available to help with different study strategies. Sign up today and start learning smarter!</div>",
                unsafe_allow_html=True)
    st.divider()
    st.markdown(f"""
        ### :page_with_curl: Support & Feedback
        - To reach me, Lets chat on:
        - {render_svg(tw_svg)} [Twitter](https://twitter.com/DFajemila)
        - {render_svg(lk_svg)} [LinkedIn](https://www.linkedin.com/in/fajemila-oluwadunsin-36332a18a/)
    """, unsafe_allow_html=True)
    st.divider()

def chat_pdf():

    c1, c2 = st.columns([1,3])
    with c1, c2:
        # radio to select document type either text or file
        doc_type = c1.radio(
            label="File",
            options=("File", "Text"),
        )
        match doc_type:
            case "Text":
                # text input to get text from user with c2
                text_input = c2.text_area(label="Paste Document Text", key="input_text")
            case _:
                # Get the list of file names from session state
                file_names = [file.name for file in st.session_state.pdf_files]
                # Display a multiselect widget with file names as options
                selected_files = c2.multiselect(
                    "Select the files you want to access", options=file_names
                )
                
    if st.button("Process"):
        with st.spinner("Processing"):
            # check if doc_type is file of text
            if doc_type == "File":
                if selected_files:
                    # Loop through the selected file names
                    for file_name in selected_files:
                        # Find the corresponding file object in session state
                        file = next(file for file in st.session_state.pdf_files if file.name == file_name)
                        # Check the file type and process it with different processor
                        raw_text = load_file(file)
                else:
                    # Display a message
                    st.write("No files selected yet.")
                    # get pdf text
                    # raw_text = get_pdf_text(file_input)
            else:
                raw_text = text_input
        
            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            
            # create vector store
            vectorstore = get_vectorstore(text_chunks)

        # create conversation_chain
        st.session_state.conversation = get_conversation_chain(vectorstore)

    user_question = st.text_input("Ask a question about your materials:")
    msgchat("Hello robot!",is_user=True)
    msgchat("Hello human!, Ask me questions from the materials you have uploaded!.")
    if user_question:
        handle_user_input(user_question)


def Generate_Quest():
    c1, c2 = st.columns([1,3])
    with c1, c2:
        # radio to select document type either text or file
        doc_type = c1.radio(
            label="File",
            options=("File", "Text"),
            horizontal=True,
        )
        match doc_type:
            case "Text":
                # text input to get text from user with c2
                text_input = c2.text_area(label="Paste Document Text", key="input_text")
            case "File":
                file_names = [file.name for file in st.session_state.pdf_files]
                file_input = c2.selectbox(
                        "Select file then click'Generate MCQ'", options=file_names)
    # to generate questions that can be answered in phrases        
    if st.button("Generate Questions"):
        with st.spinner("Generating Questions"):
            # check if doc_type is pdf of text
            if doc_type == "File":
                if file_input:
                    # Loop through the selected file names
                    for file in st.session_state.pdf_files:
                        if file.name == file_input:
                            raw_text = load_file(file)
                else:
                    st.write("No files selected yet.")

            else:
                raw_text = text_input     
        # get the text chunks
        text_chunks = get_text_chunks(raw_text, overlap=20, chunk_size=300)
        # for each text chunk, generate a question if dictionary is not False
        for text_chunk in text_chunks:
            question_generator = QuestionGenerator(text_chunk)
            question_dict = question_generator.generate_quest()
            if question_dict is not False:
                try:
                    st.session_state.quests.append(question_dict['question'])
                    st.session_state.ans.append(question_dict['answer'])
                except:
                    st.session_state.quests = [question_dict['question']]
                    st.session_state.ans = [question_dict['answer']]
        st.session_state.qtype = "QA"

    if st.button("Generate MCQ"):
        with st.spinner("Generating MCQ"):
            # check if doc_type is pdf of text
            if doc_type == "File":
                if file_input:
                    # Loop through the selected file names
                    print(file_input)
                    print(st.session_state.pdf_files)
                    for file in st.session_state.pdf_files:
                        print(file_input)
                        if file.name == file_input:
                            raw_text = load_file(file)
                else:
                    st.write("No files selected yet.")

            else:
                raw_text = text_input     
        # get the text chunks
        text_chunks = get_text_chunks(raw_text, overlap=20, chunk_size=1000)
        # for each text chunk, generate a question if dictionary is not False
        for text_chunk in text_chunks:
            question_generator = QuestionGenerator(text_chunk)
            question_dict = question_generator.generate_question()
            if question_dict is not False:
                try:
                        
                    st.session_state.questions.append(question_dict['question'])
                    st.session_state.options.append(question_dict['options'].values())
                    st.session_state.answers.append(question_dict['options'][question_dict['answer']])
                except:
                    st.session_state.questions = [question_dict['question']]
                    st.session_state.options = [question_dict['options'].values()]
                    st.session_state.answers = [question_dict['options'][question_dict['answer']]]
        st.session_state.qtype = "MCQ"

    question_form(st.session_state.qtype)




def question_form(qtype):
    # Create a form for the questions
    if qtype == "QA":
        try:
            with st.form(key="qa_form"):
                # Loop through the questions and choices
                for i, (question, answer) in enumerate(zip(st.session_state.quests, st.session_state.ans)):
                    # Display the question using write
                    st.write(f"**{i+1}. {question}**")
                    # create a text input for the answer
                    user_answer = st.text_input("", key=f"question{i}")

                # create a submit button at the last question
                submitted = st.form_submit_button("Submit")

                # calculate the score if the form is submitted
                if submitted:
                    print(st.session_state.quests)
                    print(st.session_state.ans)
                    # Calculate the score using list comprehension
                    score = sum(evaluator(st.session_state.ans[i], st.session_state[f"question{i}"]) for i in range(len(st.session_state.quests)))
                    
                    # score = sum(st.session_state[f"question{i}"] == st.session_state.ans[i] for i in range(len(st.session_state.quests)))
                    # Display the score using header
                    st.header(f"**Your score: {score}/{len(st.session_state.quests*3)}**")
                    # Loop through the questions and answers again
                    for i, (question, answer) in enumerate(zip(st.session_state.quests, st.session_state.ans)):
                        # Get the user answer from session state
                        user_answer = st.session_state[f"question{i}"]
                        user_score = evaluator(answer, user_answer)
                        # Display the question and answer using markdown
                        st.markdown(f"**{i+1}. {question}**")
                        st.markdown(f"- Correct answer: **{answer}**")
                        # Check if the user answer is correct or wrong
                        if user_score == 3:
                            # Display the user answer in green
                            st.markdown(f"- Your answer: <span style='color:green'>{user_answer}</span>", unsafe_allow_html=True)
                            st.markdown(f"- Your score: <span style='color:green'>{user_score}</span>", unsafe_allow_html=True)
                        elif user_score == 2:
                            st.markdown(f"- Your answer: <span style='color:orange'>{user_answer}</span>", unsafe_allow_html=True)
                            st.markdown(f"- Your score: <span style='color:orange'>{user_score}</span>", unsafe_allow_html=True)
                        elif user_score == 1:
                            st.markdown(f"- Your answer: <span style='color:yellow'>{user_answer}</span>", unsafe_allow_html=True)
                            st.markdown(f"- Your score: <span style='color:yellow'>{user_score}</span>", unsafe_allow_html=True)
                        else:
                            # Display the user answer in red
                            st.markdown(f"- Your answer: <span style='color:red'>{user_answer}</span>", unsafe_allow_html=True)
                            st.markdown(f"- Your score: <span style='color:red'>{user_score}</span>", unsafe_allow_html=True)
        except:
            pass
    else:
        try:
            with st.form(key="mcq_form"):
                # Loop through the questions and choices
                for i, (question, choice) in enumerate(zip(st.session_state.questions, st.session_state.options)):
                    # Display the question using write
                    st.write(f"**{i+1}. {question}**")
                    # Create a radio button for the choices
                    user_answer = st.radio("", key=f"question{i}", options=[f"{c}" for c in choice])
                # create a submit button at the last question
                submitted = st.form_submit_button("Submit")

                # calculate the score if the form is submitted
                if submitted:
                    print(st.session_state.questions)
                    print(st.session_state.options)
                    print(st.session_state.answers)
                    # Calculate the score using list comprehension
                    score = sum(st.session_state[f"question{i}"] == st.session_state.answers[i] for i in range(len(st.session_state.questions)))
                    # Display the score using header
                    st.header(f"**Your score: {score}/{len(st.session_state.questions)}**")
                    # Loop through the questions and answers again
                    for i, (question, answer) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
                        # Get the user answer from session state
                        user_answer = st.session_state[f"question{i}"]
                        # Display the question and answer using markdown
                        st.markdown(f"**{i+1}. {question}**")
                        st.markdown(f"- Correct answer: **{answer}**")
                        # Check if the user answer is correct or wrong
                        if user_answer == answer:
                            # Display the user answer in green
                            st.markdown(f"- Your answer: <span style='color:green'>{user_answer}</span>", unsafe_allow_html=True)
                        else:
                            # Display the user answer in red
                            st.markdown(f"- Your answer: <span style='color:red'>{user_answer}</span>", unsafe_allow_html=True)
        except:
            pass
            

    

def rephrase():
    text_input = st.text_area(label="Paste Document Here", key="rephrase_text")

    summary_paraphrase = st.selectbox(label="Choose what to do with your text", key="role",
                             options=("Summarize", "Correct")
    )
    if st.button("Select"):
        raw_text = text_input
        R_ephrase = Rephraser(raw_text)
        match summary_paraphrase:
            case "Summarize":
                st.write(R_ephrase.summarize())
            case "Correct":
                st.write(R_ephrase.correct())
                


def therapy():
    c1, c2 = st.columns(2)
    with c1, c2:
        input_kind = c1.radio(
            label="Input Kind",
            options=("Text", "Voice [test mode]"),
            horizontal=True,
        )

    if st.session_state.user_text:
        asked, generated = st.session_state.user_text, st.session_state.bot.reply(st.session_state.user_text)
        st.session_state.history.append(asked)
        st.session_state.history.append(generated)
        print(st.session_state.history)

        
        show_conversation()
        st.session_state.user_text = ""
    get_user_input()
    show_chat_buttons()

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)


# --- LOAD CSS ---
with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def run_agi():
    st.markdown(f"<h1 style='text-align: center;'>Estudiar.ai</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{PAGE_TITLE}</h3>", unsafe_allow_html=True)
    selected_footer = option_menu(
        menu_title=None,
        options=[
            "Info",
            "AskDOC",
            "Q&A",
            "Construct",
            "StudyPal"
        ],
        icons=["info-circle", "book", "question-square-fill", "text-paragraph", "chat-square-text" ],  # https://icons.getbootstrap.com/
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles=HEADER_STYLES
    )
    with st.sidebar:
        uploaded_files = st.file_uploader(
                            "Upload All Your PDF, DOCX or TXT files only", type=["pdf", "docx", "txt"], accept_multiple_files=True
                        )
        # Check if any files are uploaded
        if uploaded_files:
            # Assign the list of files to session state
            st.session_state.pdf_files = uploaded_files
            # Display the number of files uploaded
            st.write(f"You have uploaded {len(uploaded_files)} files.")
        else:
            # Initialize an empty list for session state
            st.session_state.pdf_files = []
        
        reset_conv = st.button("Reset QA", on_click=reset_QA)

    match selected_footer:
        case "AskDOC":
            chat_pdf()
        case "Q&A":
            Generate_Quest()
        case "Construct":
            rephrase()
        case "StudyPal":
            therapy()
        case _:
            img_dir_path = os.path.join(img_dir, get_random_img(get_files_in_dir(img_dir)))
            st.image(img_dir_path)
            show_info(tg_svg)


if __name__ == "__main__":
    run_agi()