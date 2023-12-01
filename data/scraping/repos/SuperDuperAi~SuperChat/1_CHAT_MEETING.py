import time
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from runtime import model

st.title("SuperChat Ai with Meeting transcript")
st.markdown(
    "**Chat with Claude v2 on Bedrock (100k context)**")

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
    "Upload your pdfs here and click on 'Process'", accept_multiple_files=True, type=['txt'])

if pdf_docs and st.session_state['doc_pdf'] == "":
    with (st.spinner('Processing')):
        text = ""
        for pdf in pdf_docs:
            # pdf_reader = PdfReader(pdf)
            # for page in pdf_reader.pages:
            #     text += page.extract_text()

            text += pdf.read().decode("utf-8")

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

        prompt_template =f"""I'm going to provide you with a transcript or document from a recent webinar or meeting. I'd like you to create an extensive long-read article suitable for blog publication based on this information. Please adhere to the following sections and guidelines in your response:

Summary:

Key Takeaways and Challenges:
a. Core Concepts: Identify and discuss the central ideas and challenges covered in the webinar.
b. Noteworthy Statements and Data Points: List any compelling arguments, statistics, or quotes.
c. Key Participants: Identify the main speakers and elaborate on their roles and points of view.
d. Related Topics and Influencers: Suggest tags for associating this content with other relevant webinars, articles, or experts in the field.

Structural Analysis According to Three Segments:
a. Introduction: Summarize the opening segment, including the setting, main speakers, and the primary topics to be covered.
b. Discussion and Debate: Describe the core discussions, disagreements, and breakthrough moments.
c. Conclusion and Takeaways: Sum up how the webinar ended, including any conclusions, action items, or unresolved questions.

Content Evaluation:
a. Sentiment Analysis: Is the overall sentiment positive, negative, or neutral? Provide textual evidence. Rate the sentiment on a scale of -1 to 1.
b. Content Safety Check: Examine for any content promoting violence, discrimination, or hatred towards individuals/groups and provide supporting excerpts.

Speaker Metrics:
Clarity Score: On a scale of 1 to 10, rate how clearly the speaker articulated their points.
Engagement Level: On a scale of 1 to 10, rate how well the speaker engaged the audience.
Subject Mastery: On a scale of 1 to 10, rate the speaker's expertise in the subject matter.
Pace and Timing: On a scale of 1 to 10, rate the appropriateness of the speaker's pace and use of time.
Audience Interaction: On a scale of 1 to 10, rate the speaker's ability to interact with and respond to the audience.
Visual Aids: On a scale of 1 to 10, rate the effectiveness of any visual aids, slides, or props used.

Meeting Metrics:
Agenda Adherence: On a scale of 1 to 10, rate how closely the meeting stuck to its intended agenda.
Content Relevance: On a scale of 1 to 10, rate the relevance of the content presented to the stated purpose of the meeting.
Collaboration Quality: On a scale of 1 to 10, rate the quality of discussions, debates, and collaborations.
Outcome Achievement: On a scale of 1 to 10, rate how well the meeting achieved its intended outcomes or objectives.
Duration Appropriateness: On a scale of 1 to 10, rate whether the meeting duration was appropriate for its content and objectives.
Technical Execution: On a scale of 1 to 10, rate the quality of the audio, video, and any other technical aspects.

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
