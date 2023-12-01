import openai
import streamlit as st
import sqlite3 as sql
from googletrans import LANGUAGES
from src.translate import translate_lang
from src.camera import camera_recognition
from src.utils import favicon

openai.api_key = st.secrets["keys"]["OPENAI_KEY"]

st.set_page_config(
    page_title="Self.Translate",
    page_icon=favicon(),
    layout="centered",
    initial_sidebar_state="expanded"
)

def clear_table():
    database, cursor = connect_database()
    cursor = database.cursor()
    cursor.execute("DELETE FROM translations")
    database.commit()
    database.close()

def connect_database():
    database = sql.connect("translation_log.db")
    cursor = database.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS translations
                (input, output)''')
    database.commit()

    return database, cursor

def regular_translate():
    languages = {v: k for k, v in LANGUAGES.items()}

    lang_names = sorted(list(languages.keys()))

    target_lang = st.selectbox("Select target language", options=lang_names)

    user_input = st.text_area(label=f"Input text: ", value="", max_chars=4000, disabled=False, placeholder="enter text", label_visibility="visible")

    if not user_input == "":
        query = user_input
        language = languages.get(target_lang)
        response, lang_name = translate_lang(query, language)
        st.text_area(label=f"Detected Language: {lang_name}", value=f"{query}", disabled=True, label_visibility="visible")
        st.text_area(label=f"Language Output: {target_lang}", value=f"{response}", disabled=True, label_visibility="visible")
        return query, response
    return None, None

page = st.sidebar.selectbox(label='Options', options=('Translate', 'Sign', 'Lesson'))

col1, col2 = st.columns([4,2.2])

with col1:
    if page == 'Translate':
        query, response = regular_translate()
        if st.sidebar.button("Clear Translation Log"):
            clear_table()

    elif page == "Sign":
        col1.markdown(f'<h2> self.<span style="background-color:#002b36;color:#6dac32;font-size:46px;border-radius:100%;">{"sign"}</span> </h2>', unsafe_allow_html=True)
        camera_recognition()
        st.write("Work in Progress!")
        st.write("""
        So far: Uses webcam to detect hands.
        Later: Implement a model to convert it to sign language.
        """)

    elif page == "Lesson":
        col1.markdown(f'<h2> self.<span style="background-color:#002b36;color:#6dac32;font-size:46px;border-radius:100%;">{"learn"}</span> </h2>', unsafe_allow_html=True)
        level = st.radio("Level", options=("Beginner", "Intermediate", "Advanced"))
        language = st.selectbox("Select target language", options=LANGUAGES.values())

        if(level == "Beginner"):
            prompt = f"I'm a {level} learner. Can you give me a lesson plan on how to learn {language}?"
        else:
            prompt = f"I'm an {level} learner. Can you give me a lesson plan on how to learn {language}?"
            
        st.write("Prompt: "+ prompt)

        if st.button("Generate me a lesson plan"):
            with st.spinner("Generating lesson plan..."):
                response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.5,
                max_tokens=3000,
                n=1,
                stop=None,
            )
            response = response.choices[0].text.strip().replace("please?", "")
            st.write("This can be modified again! Just click the button above to generate a new lesson plan.")
            st.text_area(label="",value=f"{response}", height=400 , disabled=False, label_visibility="visible")

with col2:
    # this needs to change so that the translation log refreshes when the entire web app refreshes
    # for now it only outputs the translated output when the user inputs something
    # this is because we are clearning the table everytime
    if page == 'Translate':
        database, cursor = connect_database()
        st.write("Translation Log:")
        st.write("Input -> Output")
        if query != None and response != None:
            cursor.execute("INSERT INTO translations VALUES (?, ?)", (query, response))
            database.commit()
            cursor.execute("SELECT * FROM translations")
            for row in cursor.fetchall():
                st.write(f"{row[0]} -> {row[1]}")
            database.close()

