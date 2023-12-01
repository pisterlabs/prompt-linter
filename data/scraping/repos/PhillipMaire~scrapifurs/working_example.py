
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd
from dotenv import load_dotenv
import os
import openai

from bs4 import BeautifulSoup
import re
import time
import pickle
import numpy as np 


from scrapifurs import utils
from scrapifurs.GPTinstructions import GPTinstructions

# setup basic variable as dict 
info_dict = {'init_url':'https://www.linkedin.com/',
             'save_password_dir':'/Users/phil/Dropbox/GITHUB/DATA/scrapifurs/saved_cookies/',
             'start_url':'https://www.linkedin.com/search/results/people/?keywords=data%20scientist&origin=CLUSTER_EXPANSION&sid=fRq'}
info_dict['full_cookies_save_path'] = info_dict['save_password_dir']+os.sep+"linkedin_cookies.pkl"


# setup API key for chatGPT 
load_dotenv()  # take environment variables from .env.
os.environ["OPENAI_API_KEY"]  = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ["OPENAI_API_KEY"]


#init chrome 
chrome_options = Options()
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
driver = webdriver.Chrome(options=chrome_options)

driver.get(info_dict['init_url'])
time.sleep(2)
driver.get(info_dict['init_url'])
time.sleep(2)


# Load cookies if they exist
try:
    cookies = pickle.load(open(info_dict['full_cookies_save_path'], "rb"))
    for cookie in cookies:
        driver.add_cookie(cookie)
    driver.refresh()
    assert(not not cookies)# if empty try a different method
except:
    print("No cookies found. Manual login required.")
    # If not logged in
    input('Please login and press Enter to continue...')
    pickle.dump(driver.get_cookies(), open(info_dict['full_cookies_save_path'], "wb")) # save cookies after login
    

input('''set zoom to 25% for winow to see all website data that it needs, press enter to continue''')

# allow you to save instructions to append ot the beginning of a GPT command based on the txt files in 
# the data directory (or set a custom directory). modular for the many different commands and instructions 
# we will need. 
instructions = GPTinstructions()

# # Print all instructions keys
# instructions.print_instructions()

tmp1 = instructions.get_instruction("linkedinSearchExtractNamesDF")


text_finder = utils.StringSectionExtractor()
text_finder.add_start_rule('search result pages', False)
text_finder.add_end_rule('Page \d+ of \d+', True)
text_finder.add_end_rule("these results helpful", False)
text_finder.add_end_rule("messaging overlay", False)



driver.get(info_dict['start_url'])
all_text = []


n_times = np.random.uniform(12, 45, 60)
for k in n_times:
    time.sleep(k)
    url_text = utils.get_lxml_text(driver, remove_empty_lines=True)
    text_data = text_finder.extract(url_text)
    all_text.append(text_data)
    utils.click_next_button(driver)
    
f_name = '/Users/phil/Dropbox/GITHUB/DATA/scrapifurs/linked_in_search_saves/31_to_100_pages.pkl'
with open(f_name, 'wb') as f:
    pickle.dump(all_text, f)



for k in all_text:
    print(len(k))









