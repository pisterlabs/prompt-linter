import os
#from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import PyPDF2
from PIL import Image
import glob
from gtts import gTTS
import os
import time

try:
    os.mkdir("temp")
except:
    pass


st.title('Oráculo de  q\'ij Tikal. 💬')

image = Image.open('chat.png')

st.image(image, width=600)

st.write(' Al ingresar al recinto del oráculo de q\'ij Tikal, una sinfonía de luces parpadeantes y hologramas danzantes recibe a los visitantes.'
         ' El zumbido constante de cables entrelazados crea una atmósfera casi hipnótica, mientras que los murmullos de datos fluyen como '
         ' corrientes invisibles por los rincones del recinto. El oráculo, envuelto en una amalgama de tecnología ancestral y moderna, se yergue'
         ' como un guardián de la sabiduría oculta, sus ojos luminosos fijos en aquellos que buscan respuestas en el torrente de información que fluye a su alrededor.')

ke = st.text_input('Escribe el código Secreto que te permite acceso al conocimiento de Ruway.')
try:
    os.environ['OPENAI_API_KEY'] = ke
    
    pdfFileObj = open('Ruway.pdf', 'rb')
     
    # creating a pdf reader object
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    
    
        # upload file
    #pdf = st.file_uploader("Carga el archivo PDF", type="pdf")
    
       # extract the text
    #if pdf is not None:
    from langchain.text_splitter import CharacterTextSplitter
     #pdf_reader = PdfReader(pdf)
    pdf_reader  = PyPDF2.PdfReader(pdfFileObj)
    text = ""
    for page in pdf_reader.pages:
             text += page.extract_text()
    
       # split into chunks
    text_splitter = CharacterTextSplitter(separator="\n",chunk_size=500,chunk_overlap=20,length_function=len)
    chunks = text_splitter.split_text(text)
    
    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    # show user input
    st.subheader("Has sido autorizado, Realiza la pregunta")
    user_question = st.text_input(" ")
    if user_question:
            docs = knowledge_base.similarity_search(user_question)
    
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
              response = chain.run(input_documents=docs, question=user_question)
              print(cb)
            st.write(response)
    
            def text_to_speech(text, tld):
                    
                    tts = gTTS(response,"es", tld , slow=False)
                    try:
                        my_file_name = text[0:20]
                    except:
                        my_file_name = "audio"
                    tts.save(f"temp/{my_file_name}.mp3")
                    return my_file_name, text
    
        
            if st.button("Escuchar"):
              result, output_text = text_to_speech(response, 'es')
              audio_file = open(f"temp/{result}.mp3", "rb")
              audio_bytes = audio_file.read()
              st.markdown(f"## Escucha:")
              st.audio(audio_bytes, format="audio/mp3", start_time=0)
    
  
    
                
              def remove_files(n):
                    mp3_files = glob.glob("temp/*mp3")
                    if len(mp3_files) != 0:
                        now = time.time()
                        n_days = n * 86400
                        for f in mp3_files:
                            if os.stat(f).st_mtime < now - n_days:
                                os.remove(f)
                                print("Deleted ", f)
                
                
              remove_files(7)

except:    
         pass
image = Image.open("logo.png")
st.image(image)
