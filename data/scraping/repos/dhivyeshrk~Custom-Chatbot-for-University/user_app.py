# user_app.py
import streamlit as st
import os
from langchain.vectorstores import Chroma
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer, util
import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import sqlite3

def generate_email_content(subject):
    import openai

    openai.api_key = ''  # Replace with your actual API key

    prompt = f"Compose a formal email message regarding {subject}. Avoid using placeholders like [Recipient's Name] or [Mode of Payment]."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates formal email content who writes emails of approx. 30 words"},
            {"role": "user", "content": prompt},
        ]
    )
    content = response.choices[0].message['content']
    return content

def generate_email_subject(subject):
    import openai

    openai.api_key = ''  # Replace with your actual API key

    prompt = f"Generate a formal email subject regarding {subject}. Avoid using placeholders like [Recipient's Name] or [Mode of Payment]."
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=15,
        n=1,
        stop=["\n"]
    )
    subject = response.choices[0].text.strip()
    return subject



def sendEmail(emailidrec, sub, emailidsend="", generate=True):
    import smtplib
    from email.message import EmailMessage

    email_address = ""
    email_password = ""
    msg = EmailMessage()
    msg['Subject'] = sub
    msg['From'] = emailidsend
    msg['To'] = emailidrec

    if generate:
        msg.set_content(generate_email_content(sub))
    else:
        if sub == 'MEDICAL_LEAVE.TXT':
            msg.set_content("Greetings Sir, \n this is to inform you about my inability to attend classes owing to. "
                            "medical reasons and i want to convey a hello ")
        elif sub == 'leave_form.txt':
            msg.set_content("Greetings Sir, \n this is to inform you about my inability to attend classes owing to emergency reasons.")

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_address, email_password)
            smtp.send_message(msg)
    except Exception as e:
        print(f"Error: {e}")



os.environ['OPENAI_API_KEY'] = ''

# Set Streamlit page configuration
st.set_page_config(
    page_title='User - LLM QA File',
    page_icon=":information_desk_person:",
    menu_items=None
)

# Define Razer-themed background styling
st.markdown(
    """
    <style>
    body {
        background-image: url('https://your-image-url-here.com');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k}, metadata_fields=['purpose'])
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer

def get_similar_use_case(query: str):
    root_dir = r"Similar_check_TextFiles"
    file_names = os.listdir(root_dir)
    allscore = []
    for file_name in file_names:
        file_path = os.path.join(root_dir, file_name)
        with open(f"{file_path}", 'r') as f:
            text = f.read()
        sentences = [sentence.strip() for sentence in re.split(r'[.!?]', text) if sentence.strip()]
        mscore = -10
        for sen in sentences:
            embed1 = model.encode(sen, convert_to_tensor=True)
            embed2 = model.encode(query, convert_to_tensor=True)
            cosine_score = util.pytorch_cos_sim(embed2, embed1)
            mscore = max(mscore,cosine_score)
        allscore.append([mscore,file_name])
    temp = [i for i,j in allscore]
    result = [[msc, fname] for msc, fname in allscore if msc == max(temp)]
    return result[0]

# Main Streamlit app for user side
if __name__ == "__main__":
    st.subheader('LLM Chatbot Application :information_desk_person:')

    vector_store = Chroma(persist_directory=r"C:\Users\DELL\DataspellProjects\Chatbot_Test_2\ChatWithDocument\Data_Files_"
                                            r"2",
                          embedding_function=OpenAIEmbeddings())

    st.session_state.vs = vector_store
    q = st.text_input('Ask a question about the content of your file:')
    if q and get_similar_use_case(q)[0] > 0.5:
        email_type = get_similar_use_case(q)[1]

        yesno1 = st.text_input("Do you want to send an email - Yes or No")
        if yesno1.lower() == 'yes':

            conn = sqlite3.connect('Mail_Feature.db')
            cursor = conn.cursor()

            cursor.execute(f"Select REQUIRED_PARAMETER1,REQUIRED_PARAMETER2,REQUIRED_PARAMETER3 from Level1 where TYPE_OF_QUERY = '{email_type}'")
            Required_parameters = cursor.fetchall()

            yesno2 = st.text_input(f"Do you want to continue with {email_type}")
            if yesno2.lower() == 'yes':
                name = st.text_input("Enter name")
                roll = st.text_input("Enter roll")
                details = st.text_input("Enter details to include")
                cursor.execute(f"SELECT DESTINATION_MAIL1, DESTINATION_MAIL2 FROM LEVEL1 WHERE TYPE_OF_QUERY = '{email_type}'")
                destination_mail = cursor.fetchone()
                fac_mail = ""
                try:
                    if destination_mail[0] == 'FACULTY':
                        fac_mail = st.text_input("Enter faculty emails: ")
                except:
                    print()
                entered_password = st.text_input('Enter Password')
                cursor.execute(f"SELECT PASSWORD FROM LEVEL2 WHERE ROLLNO = '{roll}'")
                original_password = cursor.fetchone()
                try:
                    print("Original passowrd is " + original_password[0])
                except:
                    print()

                if original_password is not None:  # Check if original_password is not None
                    original_password = original_password[0]
                try:
                    if entered_password == original_password:
                        st.write("Sending ...")
                        sendEmail(fac_mail, sub=f'{details}')
                        if destination_mail[1] != None:
                            sendEmail(destination_mail[1], sub = '')
                        cursor.execute(f"INSERT INTO LEVEL3(NAME, ROLLNO, DESTINATION_MAIL1, TYPE_OF_QUERY) VALUES('{name}', '{roll}', '{destination_mail[0]}', '{email_type}')")
                        conn.commit()
                        st.text("EMail sent successfully")
                    elif entered_password != "":
                        st.text("Password is wrong. Try Again")
                except:
                    print()
                # UPLOAD PICTURE and other operations

            elif yesno2.lower() != "":
                mail_types = ["Medical Certificate", "Fee Information", "Leave Booking", "Open"]
                selected_mail_type = st.radio("Select the type of mail:", mail_types)
                if selected_mail_type and selected_mail_type != "Open":
                    cursor.execute(f"Select destination_mail1, destination_mail2 from Level1 where TYPE_OF_QUERY = '{selected_mail_type}'")
                    destination_mail = cursor.fetchone()

                    name = st.text_input('Enter name')
                    roll = st.text_input('Enter roll')
                    details = st.text_input("Enter details")

                    entered_password = st.text_input('Enter Password')
                    cursor.execute(f"SELECT PASSWORD FROM LEVEL2 WHERE ROLLNO = '{roll}'")

                    try:
                        original_password = cursor.fetchone()
                        if original_password is not None:
                            original_password = original_password[0]
                    except:
                        print()
                    try:
                        if entered_password == original_password:
                            st.text("Sending ...")
                            sendEmail(destination_mail[0], sub=f'{details}')
                            cursor.execute(f"INSERT INTO LEVEL3(Name, ROLLNO, Destination_mail1, type_of_query) VALUES('{name}', '{roll}', '{destination_mail[0]}', '{selected_mail_type}')")
                            conn.commit()
                            st.text("Email Sent successfully")
                        elif entered_password != "":
                            st.text("Password is wrong. Try Again")
                    except:
                        print()
                elif selected_mail_type != "":
                    name = st.text_input("Enter name")
                    roll = st.text_input("Enter roll")
                    mail_subject_custom = st.text_input("Give a note on the type of mail you want to send")
                    destination_mail = st.text_input("Give the destination mail address")
                    entered_password = st.text_input("Enter password")
                    cursor.execute(f"SELECT PASSWORD FROM LEVEL2 WHERE ROLLNO = '{roll}'")
                    try:
                        original_password = cursor.fetchone()
                        if original_password is not None:
                            original_password = original_password[0]
                    except:
                        print()

                    if entered_password == original_password:
                        st.text("Sending mail")
                        sendEmail(destination_mail, mail_subject_custom, generate = True)
                        cursor.execute(f"INSERT INTO LEVEL3(NAME, ROLLNO, DESTINATION_MAIL1, TYPE_OF_QUERY) VALUES('{name}', '{roll}', '{destination_mail[0]}', '{selected_mail_type}')")
                        conn.commit()
                        st.text("Mail Sent")
                    elif entered_password is not None:
                        st.text("Password is wrong. Try Again.")

            conn.close()

            # Now you have the data from the database and can use it further in your application
        # else:
        #     passon = True


# if the user entered a question and hit enter
    if q and 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
        vector_store = st.session_state.vs
        answer = ask_and_get_answer(vector_store, q)

        # text area widget for the LLM answer

        st.text_area('LLM Answer: ', value=answer)

        st.divider()

        # if there's no chat history in the session state, create it
        if 'history' not in st.session_state:
            st.session_state.history = ''

        # the current question and answer
        value = f'Q: {q} \nA: {answer}'

        st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
        h = st.session_state.history

        # text area widget for the chat history
        st.text_area(label='Chat History', value=h, key='history', height=400)

# run the app: streamlit run ./ChatWithDocuments.py
