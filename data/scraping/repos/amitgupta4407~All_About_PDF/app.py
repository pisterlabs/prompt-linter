import os
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def ChatPDF(text):
    # st.write(text)
    
    #split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    # st.write(chunks)
    # creating embeddings

    OPENAI_API_KEY = st.text_input("OPENAI API KEY", type = "password")
    if OPENAI_API_KEY:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        # st.write("Embedding Created")
        # st.write(embeddings)
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        st.write("Knowledge Base created ")
        #show user input

        def ask_question(i=0):
            user_question = st.text_input("Ask a question about your PDF?",key = i)
            if user_question:
                docs = knowledge_base.similarity_search(user_question)
                # st.write(docs)

                llm = OpenAI(openai_api_key=OPENAI_API_KEY)
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    print(cb)
                st.write(response)
                ask_question(i+1)
                
        ask_question()

def main():
    st.set_page_config(page_title="Ask ur PDF",
                       page_icon="📄")

    hide_st_style = """
            <style>
            #mainMenue {visibility: hidden;}
            footer {visibility: hidden;}
            #header {visibility: hidden;}
            </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # st.write(st.set_page_config)
    st.header("Ask your PDF 🤔💭")
    
    #uploading file
    pdf = st.file_uploader("Upload your PDF ", type="pdf")

    # extract the text
    if pdf is not None:
        option = st.selectbox("What you want to do with PDF📜", [
            "Meta Data📂",
            "Extract Raw Text📄",
            "Extract Links🔗",
            "Extract Images🖼️",
            "Make PDF password protected🔐",
            "PDF Annotation📝",
            "ChatPDF💬"
            ])
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        if option == "Meta Data📂":
            st.write(pdf_reader.metadata)
        elif option == "Make PDF password protected🔐":
            pswd = st.text_input("Enter yourpass word", type="password")
            if pswd:
                with st.spinner("Encrypting..."):
                    pdf_writer = PdfWriter()
                    for page_num in range(len(pdf_reader.pages)):
                        pdf_writer.add_page(pdf_reader.pages[page_num])
                        
                    pdf_writer.encrypt(pswd)
                    with open(f"{pdf.name.split('.')[0]}_encrypted.pdf", "wb") as f:
                        pdf_writer.write(f)

                    st.success("Encryption Successful!")
                    st.download_button(
                        label="Download Encrypted PDF",
                        data=open(f"{pdf.name.split('.')[0]}_encrypted.pdf", "rb").read(),
                        file_name=f"{pdf.name.split('.')[0]}_encrypted.pdf",
                        mime="application/octet-stream",
                    )
                    try:
                        os.remove(f"{pdf.name.split('.')[0]}_encrypted.pdf")
                    except: pass
        elif option == "Extract Raw Text📄":
            st.write(text)
        elif option == "Extract Links🔗":
            for page in pdf_reader.pages:
                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        subtype = annot.get_object()["/Subtype"]
                        if subtype == "/Link":
                            try:
                                st.write(annot.get_object()["/A"]["/URI"])
                            except: pass
        elif option == "Extract Images🖼️":
            for page in pdf_reader.pages:
                try:
                    for img in page.images:
                        st.write(img.name)
                        st.image(img.data)
                except: pass
        elif option == "PDF Annotation📝":
            for page in pdf_reader.pages:
                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        obj = annot.get_object()
                        st.write(obj)
                        st.write("***********")
                        annotation = {"subtype": obj["/Subtype"], "location": obj["/Rect"]}
                        st.write(annotation)
        elif option == "ChatPDF💬":
            ChatPDF(text)
    

if __name__ == "__main__":
    main()
