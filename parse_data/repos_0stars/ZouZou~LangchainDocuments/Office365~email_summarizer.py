import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.document_loaders import UnstructuredEmailLoader
import base64
import binascii
import re

def is_base64(s):
    try:
        # Attempt to decode the string as Base64
        decoded = base64.b64decode(s)
        
        # If decoding is successful, it's Base64 encoded
        return True
    except Exception as e:
        print(e)
        # If an exception is raised, it's not Base64 encoded
        return False

def isBase64(sb):
    try:
        # print(len(sb))
        # if isinstance(sb, str):
        #     # If there's any unicode here, an exception will be thrown and the function will return false
        #     sb_bytes = bytes(sb, 'ascii')
        # elif isinstance(sb, bytes):
        #     sb_bytes = sb
        # else:
        #     raise ValueError("Argument must be string or bytes")
        # print(base64.b64decode(sb))
        return base64.b64encode(base64.b64decode(sb)) == b'' + sb
    except Exception as e:
        print(e)
        if (str(e) == 'Incorrect padding'):
            # print('here')
            print(sb + '=' * (-len(sb) % 4))
            print(base64.b64encode(base64.b64decode(sb + '=' * (-len(sb) % 4))))
            return isBase64(sb + '=' * (-len(sb) % 4))
        return False

def is_base64_encoded(input_string, altchars=b'+/'):
    try:
        # Try decoding the input string as Base64
        encoded_bytes = bytes(input_string, 'utf-8')
        encoded_bytes = re.sub(
            rb'[^a-zA-Z0-9%s]+' %
            altchars, b'', encoded_bytes)

        missing_padding_length = len(encoded_bytes) % 4

        if missing_padding_length:
            encoded_bytes += b'=' * (4 - missing_padding_length)

        result = base64.b64decode(encoded_bytes, altchars)

        # result = (base64.b64decode(input_string + '=' * (-len(input_string) % 4)))
        # result = bytes(input_string, 'ascii').decode()
        # print('decoded string')
        print(result)
        # If decoding is successful, it's Base64 encoded
        return True
    except Exception as e:
        print(e)
        if (str(e) == 'Incorrect padding'):
            return is_base64_encoded(input_string + '=' * (-len(input_string) % 4))
        # If an exception is raised, it's not Base64 encoded
        return False

def decode_content(content):
    if (is_base64(content) == True):
        return str(base64.b64decode(content + '=' * (-len(content) % 4)))
    else:
        return content

def get_emails():
    kwargs = {"mode": "elements"}
    # loader = DirectoryLoader(
    #     'emails/', 
    #     glob='**/*.eml', 
    #     show_progress=True, 
    #     use_multithreading=True, 
    #     loader_cls=UnstructuredEmailLoader,
    #     loader_kwargs=kwargs
    # )
    loader = UnstructuredEmailLoader(
        # "emails/Doc384714_78718.eml",
        "emails/Doc384605_78718.eml",
        mode="elements", 
        process_attachments=True
    )    
    return loader.load() 

def display_file_code(filename):
    with open(filename, "r") as file:
        code = file.read()
    with st.expander(filename, expanded=False):
        st.code(code, language='python')


def email_summarizer(email, subject):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1, max_tokens=256)
    templates = {
        'sender': "Determine the sender from the following subject: {subject}\n\n and email:\n\n{email}\n\nSender:",
        'role': "Determine the role of the sender from the following subject: {subject}\n\n and email:\n\n{email}\n\nRole:",
        'tone': "Provide the overall tone from the following subject: {subject}\n\n and email:\n\n{email}\n\nTone:",
        'summary': "Write a brief summary from the following subject: {subject}\n\n and email:\n\n{email}\n\nSummary:",
        'spam': "Determine if the following email is spam. I am a developer dealing with new clients, bussiness connections\
                and financial transactions, a lot of links are shared, is this email spam or not:\n\\n subject: {subject}\
                \n\nemail:\n{email}\n\nIs Spam?:",
    }
    outputs = {}
    for key, template in templates.items():
        prompt = PromptTemplate(input_variables=["email","subject"], template=template)
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        output = chain.run(email=email, subject=subject)
        outputs[key] = output

    with st.expander("Email Summary", expanded=True):
        for key, output in outputs.items():
            st.subheader(f"{key.capitalize()}:")
            st.write(output, end="\n\n")
        
def main():
    st.title("AI Email Summarization Tool")
    st.header("Powered by OpenAI, Langchain, Streamlit")
    deploy_tab, code_tab= st.tabs(["Deployment", "Code"])
    email_data = get_emails()
    content = ''
    subject = ''
    for email in email_data:
        if ((email.metadata['category'] == 'NarrativeText') | (email.metadata['category'] == 'UncategorizedText')):
            if (subject == ''):
                subject = email.metadata['subject']
            # metadata = email.metadata
            content = content + decode_content(email.page_content)
    # st.write(email_data)
    with deploy_tab:
        subject = st.text_input("Subject:", subject)
        email = st.text_area("Email:", content, height=300 )
        if st.button("Summarize Email"): email_summarizer(email, subject)
    with code_tab:
        st.header("Source Code")
        display_file_code("email_summarizer.py")

if __name__ == "__main__":
    load_dotenv()
    main()