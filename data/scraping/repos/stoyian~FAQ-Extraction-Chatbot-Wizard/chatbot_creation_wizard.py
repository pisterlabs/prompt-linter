import sys
from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.request import urlopen
from transformers import GPT2TokenizerFast
from emoji import emojize
import re
import cohere
import streamlit as st
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import os 


api_key = st.secrets["COHERE_API_KEY"]
co = cohere.Client(api_key)

def _emojize(text: str) -> str:
  dic = {":quality:" : '👌',
         ":satisfied:" : '👍',
          ":+1:": '👍',
         ":moneybag:": '💰',
         ":one:": '1️⃣',
         ":dollar:":'💵',
         ":heavy_check_mark:": '✔️',
         ":thumbsup:":'👍',
         ":arrow_up:":'⬆️',
         ":chart_with_upwards_trend:":'📈',
         ":iphone:":'📱',
         ":computer:":'🖥️',
         ":sustainability:": '🌎',
         ":car:": '🏎️'
         }
  text = emojize(text)
  l = text.split(' ')
  l = list(map(lambda x: x.replace(x, dic[x] if x in dic.keys() else x), l))
  return " ".join(l)

# Token counter
def count_tokens(input: str):
    # Get GPT2 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Tokenize text
    res = tokenizer(input)['input_ids']
    # Return the length of the tokens
    return len(res)
    
# Get text from a URL
def get_content(url: str, mode: str = "all") -> str:
  # Get html from url using selenium -- this ensures we get also dynamic text
  html = get_html_chrome(url)

  # Use beautiful soup to parse the html 
  soup = BeautifulSoup(html, features="html.parser")

  # Preprocess to select text
  if mode == "all":
    # 1. kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    # 2. get text
    text = soup.get_text()
    # 3. break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # 4. break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # 5. drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    # 6. count tokens
    num_tokens = count_tokens(text)
    if num_tokens > 2048:
      mode = "p"
  if mode == "p":
    # select only p elements
    p_list = soup.find_all("p")
    text = " ".join(p.get_text().strip() for p in p_list)
    num_tokens = count_tokens(text)
    if num_tokens > 2048:
      mode = "p2"
  if mode == "p2":
    # not supported
    print('Not supported mode, please select "all" or "p".')
    text = text[:2048]
  # Print some diagnostics
  print(f'Webpage content extraction complete. Total size: {len(text)} chars.')
  #print(f" First 100 characters: {text[0:100]}')")
  return text

# Get the HTML source from a URL
def get_html_raw(url: str) -> str:
  html = urlopen(url).read()
  return html

# Get the rendered HTML from a URL as it is seen from a browswer
def get_html_chrome(url: str) -> str:
  firefoxOptions = Options()
  firefoxOptions.add_argument("--headless")
  service = Service(GeckoDriverManager().install())
  driver = webdriver.Firefox(executable_path=GeckoDriverManager().install(), options=firefoxOptions)
  driver.get(url)
  html = driver.page_source
  return html


#---------------------------------------
# Functions for the FAQ bot
#---------------------------------------


# Generate FAQ bot content 
def generate_faq_bot(url: str):
  # First try with extracting text from all html elements
  text = get_content(url, mode="all")
  second_try = False
  try:
    name = extract_company_name(text)
  except Exception as e:
    print("Debug:", sys.exc_info()[0])
    print("Debug:", e)
    print("Error: Content extraction from webpage with method 'all' not suitable.")
    second_try = True

  # Second try with extracting text from p elements only
  if (second_try):
    text = get_content(url, mode="p")
    try:
      name = extract_company_name(text)
    except Exception as e:
      print("Debug:", sys.exc_info()[0])
      print("Debug:", e)
      print("Error: Content extraction from webpage with method 'p' not suitable.")
      raise Exception("Webpage text extraction is not able to extract content of appropriate size.")
  
  # Assuming that the name extraction worked, we expect the other calls to work too.

  # Extract company sector
  sector = extract_company_sector(text)
  # Generate FAQs from extracted text
  faqs, questions_list, answers_list = generate_faqs(text)
  # Generate labels for generated FAQs
  labels, labels_list = generate_question_labels(questions_list)
  # Generate hyped versions of the generated answers
  answers_hyped_list = generate_hyped_answers(answers_list)
  faqbot_dict = {

      "questions": questions_list,
      "answers": answers_list,
      "hyped": answers_hyped_list,
      "labels": labels_list,
      "name": name,
      "sector": sector
  }
  return faqbot_dict

# Print FAQ bot content from generated dictionary 
def print_faq_bot(faqbot: dict): 
  print(f'Company name: {faqbot["name"]}')
  print(f'Company sector: {faqbot["sector"]}')
  for i in range(3):
    print(f"""FAQ #{i+1}
    Question: {faqbot["questions"][i]}
    Answer: {faqbot["answers"][i]}
    Hyped answer: {faqbot["hyped"][i]}""")

# Generate FAQs from text grabbed from a webpage 
def generate_faqs(text: str, num: str = "three", temperature: float = 0.6):
  # Use a prompt that instructs the model to generate FAQs.
  prompt = f"""Create {num} Frequently Asked Questions (FAQs) and their corresponding answer from the company description below. 
  Company description: {text}
  FAQ #1.
  Question:"""
  response = co.generate(prompt = prompt, 
                         model='xlarge-20221108',
                         max_tokens = 300,
                         temperature = temperature)
  # Extract questions and answers separately too
  faqs = f'FAQ #1.\n  Question: {response.generations[0].text}'
  questions_list = re.findall(r'Question:\s(.*)', faqs)
  questions_list = [q.strip() for q in questions_list]
  answers_list = re.findall(r'Answer:\s(.*)', faqs)
  answers_list = [a.strip() for a in answers_list]
  print("FAQ extraction complete.")
  return faqs, questions_list, answers_list

# Rewrite answers to a more hyped form 
def generate_hyped_answers(answers_list: list, temperature: float = 1.5):
  #Use a prompt that instructs model to make text more hyped
  answers_hyped_list = []
  for i in range(0,3):
    answer = answers_list[i] + "\n\n"
    prompt = f"""Rewrite the following sentence using words used mostly between younger people one the internet.
    Example:
      Sentence: We use chat bots 
      New Sentence: We use super cool bot that do all the work
      Sentence: We make software with the power of AI
      New Sentence: We make awesome software solutions using the cutting edge technology of AI
      Sentence: The company focuses on improvinig the customer experience
      New Sentence: The major focus of the company is to give customers the best experience they ever had
      Sentence: {answer}
      New Sentence : """
    response = co.generate(prompt = prompt, 
                          model= 'xlarge-20221108',
                          max_tokens = 50,
                          temperature = temperature,
                          k=0,
                          p=0.73)
                                
    answer = f'#{i+1}:{response.generations[0].text}'
    answer = re.findall(r'#[1-3]:\s[\n\n]?(.*)', answer)
    answer = " ".join(answer).strip()
    answer = _emojize(answer)
    answers_hyped_list.append(answer)
  print("Hyped answers generation complete.")
  return answers_hyped_list

# Generate labels for FAQs 
def generate_question_labels(questions_list: list, temperature: float = 0.5):
  # Use a prompt that instructs the model to generate labels for sentences.
  questions = ""
  for question in questions_list:
    questions += f"Question: {question} \n"
  prompt = f"""Classify the questions below in a single class
  {questions} Classes: """
  response = co.generate(model= 'xlarge-20221108',
                         prompt=prompt,
                         max_tokens = 50,
                         temperature = temperature,
                         p=0.7)
  labels = f'Labels:{response.generations[0].text}'
  labels_list = re.findall(r'Label\s#[1-9]:(.*)', labels)
  labels_list = [l.strip() for l in labels_list]
  print("FAQ label extraction complete.")
  return labels, labels_list

# Extract company name from text grabbed from a webpage 
def extract_company_name(text: str, temperature: float = 0.2):
  prompt = f"""Extract the company name from the following company description.
  Lets think step by step: First extract extract all names -names should be with a capitalized first letter- then, find which one refers to the company
  Desciption: {text}
  Company name: """
  response = co.generate(prompt = prompt, 
                         max_tokens = 10,
                         temperature = temperature,
                         stop_sequences = ["\n"])
  name = response.generations[0].text.strip()
  print("Company name extraction complete.")
  return name

# Extract company business sector from text grabbed from a webpage 
def extract_company_sector(text: str, temperature: float = 0.2):
  prompt = f"""Extract the business sector for the following text and claffify in in the following categories.
  The categories are:
  Aerospace, Agriculture, Automotive, Chemical,Construction, Culture, Defense, Education, Energy, Entertainment, Financial Services, Food, Healthcare,
  Hospitality, Insurance, IT, Manufacturing, Media, Pharmaceutical, Professional services, Real estate, Retail,Sports, Telecommunications, Transportation, Utilities, Unspecified

  Text: {text}. Classified as:  """
  response = co.generate(model='xlarge-20221108',
                         prompt = prompt, 
                         max_tokens = 2,
                         temperature = temperature,
                         stop_sequences = ["\n"])
  name = response.generations[0].text.strip()
  print(f"Company sector extraction complete")
  return name