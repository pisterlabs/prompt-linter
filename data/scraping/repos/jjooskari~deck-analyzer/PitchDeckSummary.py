import os
import tempfile
import openai

# Streamlit 
import streamlit as st

# Langchain imports
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback

# Import ccustom templates
from Templates import default_summary_template, get_summary_prompt_template, get_refine_prompt_template

# Import custom parser module
from SummaryParser import parse_summary

# Import drive export module
from DriveExport import export_to_drive

# Import vision analyzer module
from VisionAnalyzer import get_descriptions


def main():
    openai.api_key = os.environ['OPENAI_API_KEY']

    llm = OpenAI(temperature=0, max_tokens=-1)

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 8000,
        chunk_overlap  = 250,
        length_function = len,
    )

    # Title, caption and PDF file uploader
    st.title("Pitchdeck :rainbow[Summarizer] 1.0 ✨")
    st.caption("This app uses OpenAI's GPT-3.5 to summarize pitchdecks according to a given summary template. Copyright 2023.")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Summary template input, default template is loaded from Templates.py
    summary_template = default_summary_template()
    summary_template_input = st.text_area("Summary template", value=summary_template, height=500, max_chars=1000, placeholder="Enter a summary template")
    st.subheader("Summary")
    summary_text = st.empty()

    # Initiate session state for summary storage
    if 'summary' not in st.session_state:
        st.session_state.summary = ''
    if 'summary-json' not in st.session_state:
        st.session_state['summary-json'] = {}
    
    # If summary is not empty, display it, otherwise display placeholder
    if st.session_state.summary != '':
        summary_text.text(st.session_state.summary)
    else:
        summary_text.markdown("*Summary will appear here*")
    
    # Rest of the UI elements & holders, below the summary text
    summary_button_holder = st.empty()
    summary_button = summary_button_holder.button('Generate Summary 🪄', disabled=True)
    use_vision = st.checkbox("Use GPT-4 Vision API (costs more)")
    cancel_button_holder = st.empty()
    export_button_holder = st.empty()
    export_button = None
    message_holder = st.empty()

    pages = None

    # If PDF file is uploaded, load it and split it into pages
    if pdf_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(pdf_file.read())
                pdf_path = tmp_file.name
                
                summary_button_holder.empty()

                # Activate the summary button
                if st.session_state.summary == '':
                    summary_button = summary_button_holder.button('Generate Summary 🪄', key=1)
                else:
                    summary_button = summary_button_holder.button('Re-generate Summary 🪄', key=2)

    # If summary button is pressed, generate the summary
    if summary_button:
        if summary_template_input is not None:
            summary_template = summary_template_input
        
        summary_prompt_template = get_summary_prompt_template(summary_template)
        summary_refine_template = get_refine_prompt_template(summary_template)

        summary_prompt = PromptTemplate.from_template(summary_prompt_template)
        refine_prompt = PromptTemplate.from_template(summary_refine_template)

        if not use_vision:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            combined_content = ''.join([p.page_content for p in pages])
        else:
            # Use VisionAnalyzer to get descriptions of slides
            descriptions, cost = get_descriptions(pdf_path)
            combined_content = "\n\n".join(descriptions)
        
        texts = text_splitter.split_text(combined_content)
        docs = [Document(page_content=t) for t in texts]
        steps_n = len(texts)

        message_holder.info(f"Generating summary in {steps_n} steps...", icon="📝")
        summary_button_holder.empty()

        cancel_button = cancel_button_holder.button('Cancel')
        
        if cancel_button:
            st.stop()

        chain = load_summarize_chain(
            llm, 
            chain_type="refine", 
            question_prompt=summary_prompt, 
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
            verbose=True)
        
        with st.spinner("Generating summary..."):
            with get_openai_callback() as cost:
                summaries = chain({"input_documents": docs}, return_only_outputs=True)
                print("\nNew run:\n")
                print(cost)

        message_holder.empty()
        summary_text.empty()

        summary_text.text(summaries["output_text"])
        print(summaries["output_text"]) # Can be changed to logging later
        st.session_state.summary = summaries["output_text"]

        cancel_button_holder.empty()
        summary_button = summary_button_holder.button('Re-generate Summary 🪄', key=3)
    
    # If summary generated, display export button
    if st.session_state.summary != '':
        export_button = export_button_holder.button('Export Summary to Google Drive 📁')

    # If export button is pressed, export the summary to Google Drive with DriveExport.py
    if export_button:

        message_holder.info("Exporting to Google Drive...", icon="📁")

        print("\n\nStarting to Parse")

        parsed = parse_summary(st.session_state.summary) # To Do error handling
        st.session_state['summary-json'] = parsed
        
        print("\n\nStarting to Export")

        file_name = export_to_drive(parsed) # To Do error handling

        message_holder.empty()
        message_holder.success(f"Exported to Google Drive as: {file_name}", icon="✅")
 
if __name__ == "__main__":
    main()