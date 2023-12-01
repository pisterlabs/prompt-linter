import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Set up the Chrome WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode to avoid opening a browser window
driver_path = os.path.join(os.getcwd(), 'chromedriver.exe')
chrome_service = Service(executable_path=driver_path)
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# Get CNN homepage HTML
driver.get('https://www.cnn.com')
content = driver.page_source

# Parse CNN homepage with BeautifulSoup for headlines
soup = BeautifulSoup(content, 'html.parser')
elements = soup.find_all(class_='cd__headline-text vid-left-enabled')
elements_strings = [element.get_text(strip=True) for element in elements]
cnn_headlines = 'NewsSource1 Headlines:\n\n' + '\n\n'.join(elements_strings) + '\n\n\n\n'

# Get FOX homepage HTML
driver.get('https://www.foxnews.com')
content = driver.page_source

# Parse FOX homepage with BeautifulSoup for headlines
soup = BeautifulSoup(content, 'html.parser')
elements = soup.find_all(class_='title')
elements_strings = [element.get_text(strip=True) for element in elements]
fox_headlines = 'NewsCorp2 Headlines:\n\n' + '\n\n'.join(elements_strings)

# Combine headlines
headlines = cnn_headlines + fox_headlines

# Close the WebDriver
driver.quit()

# Create embedding and setup QA for ChatGPT
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(headlines)
os.environ['OPENAI_API_KEY'] = ''
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
vectorstore = FAISS.from_texts(texts, embeddings)
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model='gpt-3.5-turbo'), chain_type="stuff", retriever=vectorstore.as_retriever())

# Define prompt
prompt = '''
Compare the headlines of NewsSource1 and NewsCorp2.
What do they each say about Trump?
Which is more critical of Trump and why is this?
What sort of general beliefs do they have?
'''

# Send prompt to ChatGPT
response = qa.run(prompt)

# Display ChatGPT's response
print(response)
