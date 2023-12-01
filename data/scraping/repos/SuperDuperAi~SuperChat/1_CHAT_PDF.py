import time
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from runtime import model

st.title("SuperChat Ai with PDF")
st.markdown(
    "**Chat with Claude v2 on Bedrock (100k context). Get started by uploading a PDF!**")

# add to sidebar inputs max_tokens_to_sample
st.sidebar.subheader('Model parameters')

pdf_docs = None
if 'doc_pdf' not in st.session_state:
    st.session_state['doc_pdf'] = ""

instruct_value = ""
instruct_text = ""

with st.sidebar:
    st.subheader('Parameters')
    chunk_size = st.sidebar.slider('chunk_size', 0, 10000, 1000)
    pdf_chunks_limit = st.sidebar.slider('pdf_chunks_limit', 0, 95000, 90000)

pdf_docs = st.file_uploader(
    "Upload your pdfs here and click on 'Process'", accept_multiple_files=True, type=['pdf'])

if pdf_docs and st.session_state['doc_pdf'] == "":
    with st.spinner('Processing'):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        data = ""
        for chunk in chunks:
            if len(data) > 90000:
                st.warning('PDFs is big, only first >100k characters will be used')
                break
            data += chunk

        prompt_template = f"""
        I'm going to provide you with document (book) in pdf file.
        Then, I'll ask you to create an extensive long-read article suitable for blog publication based on the information. 
        Please adhere to the following sections and guidelines in your response:
    
        Literary Analysis:
        a. Main Themes and Challenges: Identify and discuss the overarching themes and problems.
        b. Engaging Theses and Quotations: List interesting theses and quotes.
        c. Principal Characters: Identify the main characters and elaborate on their roles.
        d. Inter-Textual Links: Suggest tags for associating with other literary works and authors.
    
        Episodic Description According to Three-Act Structure:
        a. Act 1 - Setup: Provide a summary of the initial act, establishing the setting, characters, and the main conflicts.
        b. Act 2 - Confrontation: Describe the events and obstacles the main characters face, leading to the climax of the story.
        c. Act 3 - Resolution: Sum up how the story concludes, including how conflicts are resolved and the state of the characters.
    
        Content Assessment:
        a. Sentiment Analysis: Determine whether the sentiment in the text is positive, negative, or neutral, providing textual evidence. Rate the sentiment on a scale of -1 to 1.
        b. Destructive Content Detection: Check for any content promoting violence, discrimination, or hatred towards individuals/groups and provide supporting excerpts.
    
        Readability Metrics:
        a. Provide the Flesch Reading Ease score, Flesch-Kincaid Grade Level, and Gunning Fog Index.
    
        Political Orientation Analysis:
        a. Identify and explain liberal or conservative values, democratic or autocratic tendencies, and militaristic or humanistic themes in the text.
        b. Summarize the political orientation and rate on a scale of -1 to 1 for each dimension (Liberal-Conservative, Democratic-Autocratic, Militaristic-Humanistic).
        
        Here is the document:
        <document>
        {data}
        </document>
    
        Result in Markdown format.
        Answer in 8000 words or less.
        
        ### Questions:
        [Provide three follow-up questions worded as if I'm asking you. 
        Format in bold as Q1, Q2, and Q3. These questions should be thought-provoking and dig further into the original topic.]

        """

        # st.info(prompt_template)
        with st.chat_message("assistant"):

            st.session_state['doc_pdf'] = data
            st.success(f'Text chunks generated, total words: {len(data)}')

        pdf_summarise = model.predict(input=prompt_template)
        st.session_state.messages.append({"role": "assistant", "content": pdf_summarise})


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize session state for chat input if it doesn't already exist

prompt_disabled = (st.session_state['doc_pdf'] == "")

if prompt := st.chat_input("What is up?", disabled=prompt_disabled):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        processed_prompt = prompt

        result = model.predict(input=prompt)

        for chunk in result:
            full_response += chunk
            time.sleep(0.01)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
