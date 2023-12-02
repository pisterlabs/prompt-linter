import uuid
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import streamlit as st
import requests
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import pdf2image
from langchain.text_splitter import CharacterTextSplitter

from api_config import API_HOST, API_PORT
from respond_beauty import make_it_beautiful

try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract

API_URLS = {
    "ChatGPT-4": API_HOST + f":{API_PORT}" + "/api/v1/prediction/bce8e1fd-cb78-4068-9822-d386d914068a"
}

MODELS = [
    "ChatGPT-4"
]

if "summarize_interface_memory" not in st.session_state:
    st.session_state.summarize_interface_memory = ConversationBufferMemory()

if "summarize_model" not in st.session_state:
    st.session_state.summarize_model = "ChatGPT-4"

if "summarize_interface_html" not in st.session_state:
    st.session_state.summarize_interface_html = False

if "summarized_pdf_file" not in st.session_state:
    st.session_state.summarized_pdf_file = str

if "summarize_interface_summarized_text" not in st.session_state:
    st.session_state.summarize_interface_summarized_text = ""


# write a new pdf file using st.session_state.summarized_text
# and return the file path

def write_pdf(text):
    new_file_name = "static/" + str(uuid.uuid4()) + ".pdf"
    st.session_state.summarized_pdf_file = new_file_name
    pdf = SimpleDocTemplate(new_file_name)

    content = []

    style = getSampleStyleSheet()
    pdfmetrics.registerFont(TTFont('Arial', 'download/arial.ttf'))

    # Check if 'BodyText' style already exists
    if 'BodyText' not in style:
        style.add(ParagraphStyle(name='BodyText', fontName='Arial', fontSize=12, leading=12))

    else:
        style['BodyText'].fontName = 'Arial'
        style['BodyText'].fontSize = 11
        style['BodyText'].leading = 16

    for line in text.split("\n"):
        content.append(Paragraph(line, style['BodyText']))

    pdf.build(content)


def read_pdfs(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        for page in pdf2image.convert_from_bytes(uploaded_file.read()):
            text += pytesseract.image_to_string(page)

    return text


def get_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2700,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    print("Count of chunks: ", len(chunks))
    return chunks


def handle_user_input(prompt):
    st.session_state.summarize_interface_memory.chat_memory.add_user_message(prompt)

    def query(payload):
        selected_api_url = API_URLS[st.session_state.summarize_model]
        print("Running query on API: ", selected_api_url)
        response = requests.post(selected_api_url, json=payload)
        return response.json()

    with st.spinner("Summarizing your content..."):
        output = query({
            "question": prompt,
        })

        st.session_state.summarize_interface_memory.chat_memory.add_ai_message(output)
        st.session_state.summarize_interface_summarized_text += output + "\n"


def handle_file_input(uploaded_file):
    with st.spinner("Reading your PDF..."):
        text = read_pdfs(uploaded_file)
        chunks = get_chunks(text)

    #  Summarize each chunk and add it to the memory
    for i, chunk in enumerate(chunks):
        print("Summarizing chunk: ", i)
        handle_user_input(chunk)

    #  Write the summarized text to a new pdf file
    write_pdf(st.session_state.summarize_interface_summarized_text)


def main():
    #  Add title and subtitle
    st.title(":orange[bit AI] 🤖")
    st.caption(
        "bitAI powered by these AI tools:"
        "OpenAI GPT-3.5-Turbo 🤖, HuggingFace 🤗, CodeLLaMa 🦙, Replicate and Streamlit of course."
    )

    st.subheader("Summarize Your Content With AI")

    with st.sidebar:
        upload_file = st.file_uploader("Upload a file to summarize", type=["pdf"], accept_multiple_files=True)
        upload_button = st.button("Summarize PDF", disabled=not upload_file)
        if upload_button:
            handle_file_input(upload_file)

        download_button = st.button("Download Summarized PDF",
                                    disabled=not st.session_state.summarize_interface_summarized_text)
        if download_button:
            write_pdf(st.session_state.summarize_interface_summarized_text)
            with open(st.session_state.summarized_pdf_file, "rb") as pdf_file:
                b64 = pdf_file.read()
                st.download_button(
                    label="Download Summarized PDF",
                    data=b64,
                    file_name="summarized.pdf",
                    mime="application/pdf",
                )

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            #  List models we can use
            st.session_state.summarize_model = st.selectbox("Select a model to use Summarize:", MODELS, )

        with col2:
            st.write('<div style="height: 27px"></div>', unsafe_allow_html=True)
            second_col1, second_col2 = st.columns([2, 1])
            with second_col1:
                clear_button = st.button("🗑️ Clear history", use_container_width=True)
                if clear_button:
                    st.session_state.summarize_interface_memory.clear()
                    st.session_state.summarize_interface_summarized_text = ""

            with second_col2:
                st.session_state.summarize_interface_html = st.toggle("HTML", value=False)

    prompt = st.chat_input("✏️ Enter your content here you want to summarize for: ")
    if prompt:
        handle_user_input(prompt)

    st.sidebar.caption('<p style="text-align: center;">Made by volkantasci</p>', unsafe_allow_html=True)

    #  Display chat history
    for message in st.session_state.summarize_interface_memory.buffer_as_messages:
        if isinstance(message, HumanMessage):
            if st.session_state.summarize_interface_html:
                with open("templates/user_message_template.html") as user_message_template:
                    new_content = make_it_beautiful(message.content)
                    html = user_message_template.read()
                    st.write(html.format(new_content), unsafe_allow_html=True)

            else:
                st.chat_message("Human", avatar="🤗").write(message.content)

        elif isinstance(message, AIMessage):
            if st.session_state.summarize_interface_html:
                with open("templates/ai_message_template.html") as ai_message_template:
                    new_content = make_it_beautiful(message.content)
                    html = ai_message_template.read()
                    st.write(html.format(new_content), unsafe_allow_html=True)

            else:
                st.chat_message("AI", avatar="🤖").write(message.content)


if __name__ == "__main__":
    main()
