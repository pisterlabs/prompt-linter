
from streamlit_option_menu import option_menu
import requests
from bs4 import BeautifulSoup
import streamlit as st
from gtts import gTTS
from io import BytesIO
import openai
from youtubesearchpython import *
import io
import requests
import PyPDF2
import langchain
langchain.verbose = False
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import time
from scripts.news import *


from deta import Deta

def db(link,vectors):
    try:
      deta = Deta(st.secrets["data_key"])
      db = deta.Base("Vectors")
      db.put({"link": link, "texts": vectors})
    except:
       st.error("Access Denied")



def pdfs(s,n):
    try:
      links=[]
      try:
          from googlesearch import search
      except ImportError:
          print("No module named 'google' found")

      query = f"{s} filetype:pdf"
      for j in search(query, tld="co.in", num=1, stop=1, pause=2):
          if ".pdf" in j:
              k = j.split("/")
              print(k[-1])
              print(j)
              
              links.append(j)
      st.markdown("SEARCHING FOR THE DOCUMENTS RELATED TO "+s)
      return links
    except:
       st.error("PDF NOT FOUND")


def pdftotxt(urls):
    try:
        
      texts=""
      for url in urls:
          
          
          st.write("PROCESSING: "+url)
          with st.spinner('Wait for it...'):
              time.sleep(5)
          st.success('Done!')
          response = requests.get(url)

          # Create a file-like object from the response content
          pdf_file = io.BytesIO(response.content)

          # Use PyPDF2 to load and work with the PDF document
          pdf_reader = PyPDF2.PdfReader(pdf_file)

          # Access the document information
          num_pages = len(pdf_reader.pages)
    

          # Perform further operations with the PDF document as needed
          # For example, extract text from each page

          for page in pdf_reader.pages:
              txt=page.extract_text()
              texts += txt
              #db(url,txt)
      time.sleep(.5)      
      st.info("DATA EXTRACTION DONE")
      return texts
    except:
       st.error("Converstion Failed")

        # Print the extracted text
    
def chunks(texts,q):
    try:
        
      st.info("DATA TRANFORMATION STARTED")

      st.info("TRANFORMING DATA INTO CHUNKS")
      text_splitter = CharacterTextSplitter(separator = "\n",chunk_size = 1000,chunk_overlap  = 200,
      length_function = len,)
      texts = text_splitter.split_text(texts)
      embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["api"])
      docsearchs = FAISS.from_texts(texts, embeddings)
      chain = load_qa_chain(OpenAI(openai_api_key=st.secrets["api"]), chain_type="stuff")
      with st.spinner('Wait for it...'):
          time.sleep(5)
      st.success('Done!')
      time.sleep(.5)
      st.info("DATA IS LOADED AS CHUNKS AND READY FOR ANALYTICS PURPOSE")
      query=q
      docs = docsearchs.similarity_search(query)
      title=chain.run(input_documents=docs, question="TITLE of the paper")
      query=chain.run(input_documents=docs, question=query)
      st.write(title)
      st.write(query)
      data=f"{title},{query}"
      audio_bytes=speak(data)
      st.audio(audio_bytes, format='audio/ogg')
    except:
       st.error("Error processing Text") 

  

def ai(prompt,n):
  try:
    updated_prompt = f"act as a well experienced professor and explain {prompt} as explainig to your student think step by step and give real time examples and key points but it should be below 150 words",
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=updated_prompt,
      temperature=n,
      max_tokens=150,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.6,
      stop=[" Human:", " AI:"]
    )
    data = response.choices[0].text
    st.markdown(response.choices[0].text,unsafe_allow_html=True)
    return data
  except:
     st.error("Unable Connect To The Server")  

openai.api_key = st.secrets["api"]
start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

def sql(t,q):
  try:
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=f"###{t}\n#\n### {q}\n",
      temperature=1,
      max_tokens=2000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.6,
      stop=[" Human:", " AI:"]
    )
    data = response.choices[0].text
    st.markdown(response.choices[0].text,unsafe_allow_html=True)
    return data
  except:
     st.error("Error connecting the server")

def pearson(my_string):
  try:  
    try:
      from googlesearch import search
      import re
    except ImportError:
      print("No module named 'google' found")
    query = my_string
    pattern = "(instagram|facebook|youtube|twitter|github|linkedin|scholar|hackerrank|tiktok|maps)+\.(com|edu|net|fandom)"
    for i in search(query, tld="co.in", num=20, stop=15, pause=2):
      if (re.search(pattern, i)):
        title=(i+"website name in bold")
        st.markdown(f'<a href="{i}">view more</a>', unsafe_allow_html=True)
      else:
        print("match not found")
  except:
     st.error("Service Unavailable")
            
def yt(vd):
    try:
       
      customSearch = VideosSearch(vd,limit = 20)
      for i in range(20):
          st.video(customSearch.result()['result'][i]['link'])
    except:
       st.error("Video Not Found")

def speak(text):
    mp3_fp = BytesIO()
    tts = gTTS(text, lang='en')
    tts.write_to_fp(mp3_fp)
    return mp3_fp


def pdf(s):
    try:
      try:
          from googlesearch import search
      except ImportError:
          print("No module named 'google' found")

      query = f"filetype:pdf {s}"
      for j in search(query, tld="co.in", num=10, stop=5, pause=2):
          if ".pdf" in j:
              k = j.split("/")
              for i in k:
                  if ".pdf" in i:
                      st.write(i)
              #title=ai(j+" Explain the title and content in short in this link. the title should be in bold",1)
  #             st.components.v1.iframe(j)
              st.markdown(f'<a href="{j}">DOWNLOAD</a>', unsafe_allow_html=True)
    except:
       st.error("PDF NOT Found")
def ppt(s):
    try:
        from googlesearch import search
    except ImportError:
        print("No module named 'google' found")

    query = f"filetype:ppt {s}"
   
    for j in search(query, tld="co.in", num=10, stop=5, pause=2):
        if ".ppt" in j:
            k = j.split("/")
            for i in k:
                if ".ppt" in i:
                    st.write(i)
#             st.components.v1.iframe(j)
            st.markdown(f'<a href="{j}">DOWNLOAD</a>', unsafe_allow_html=True)


def torrent_download(search):
    try:
      url = f"https://1377x.xyz/fullsearch?q={search}"
      r = requests.get(url)
      data = BeautifulSoup(r.text, "html.parser")
      links = data.find_all('a', style="font-family:tahoma;font-weight:bold;")

      torrent = []
      ogtorrent = []
      for link in links:
          st.write(link.text)
          torrent.append(f"https://ww4.1337x.buzz{link.get('href')}")
          url = f"https://ww4.1337x.buzz{link.get('href')}"
          r = requests.get(url)
          data = BeautifulSoup(r.text, "html.parser")
          links = data.find_all('a')
          for link in links:
              link = link.get('href')
              if "magnet" in str(link):
                  st.markdown(f'<a href="{str(link)}">DOWNLOAD</a>', unsafe_allow_html=True)
              if "torrents.org" in str(link):
                  ogtorrent.append(str(link))
    except:
       st.error("Course Not Found")

def display(data):
  try:
  
    st.image("images/search1.png")
    def local_css(file_name):
          with open(file_name) as f:
              st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    def remote_css(url):
          st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)
    def icon(icon_name):
          st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)
        
    local_css("style.css")
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')


    form = st.form(key='my-form')

    selected = form.text_input("", "")
    submit = form.form_submit_button("SEARCH")
    if submit:
      if "PDF" in data:
        pdf(selected)
      elif "PPT" in data:
        ppt(selected)
      elif "Courses" in data:
        st.write('''Fair Use Act Disclaimer
          This site is for educational purposes only!!
                              **FAIR USE**
        Copyright Disclaimer under section 107 of the Copyright Act 1976, allowance 
        is made for “fair use” for purposes such as criticism, comment, news reporting, teaching, 
        scholarship, education and research.Fair use is a use permitted by copyright statute that might
        otherwise be infringing. Non-profit, educational or personal use 
        tips the balance in favor of fair use. ''')
        torrent_download(selected)
        
                    

      elif "Research papers" in data:
        selected = f"{selected} research papers"
        pdf(selected)
      elif "Question Papers" in data:
        selected = f"{selected} Question Papers"
        pdf(selected)
      elif "E-BOOKS" in data:
        selected = f"{selected} BOOK"
        pdf(selected)
      elif "Hacker Rank" in data:
        st.write(f"[OPEN >](https://www.hackerrank.com/domains/{selected})")
  except:
     st.error("Service Unavailable")
      
