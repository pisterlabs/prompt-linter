from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import re
from datetime import datetime
import pandas as pd
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

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
import pandas as pd

from scrapifurs import utils
from scrapifurs.GPTinstructions import GPTinstructions

from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import linkage, leaves_list

from geopy.geocoders import Nominatim
from geopy.distance import geodesic


import plotly.express as px

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from datetime import datetime






def open_browser(info_dict):
    #init chrome 
    chrome_options = Options()
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)
    
    driver.get(info_dict['init_url'])
    time.sleep(1)
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
    time.sleep(4)
    return driver
    

def update_csv(data_df, file_name):
    """
    Update the master_jobs_applied_to.csv with the provided data DataFrame.
    
    Parameters:
    - data_df: DataFrame containing the scraped data.
    """
    
    # Enforce consistent datatypes
    data_df = data_df.astype({
        'linked_in_link': 'string',
        'job_id': np.int64,
        'job_title': 'string',
        'company_name': 'string',
        'apply_link': 'string',
        'job_location': 'string',
        'time_since_posted': 'string',
        'num_applicants': 'string',
        'salary_range': 'string',
        'job_type': 'string',
        'job_level': 'string',
        'company_info': 'string',
        'description': 'string',
        'date_number': np.int64,
        'formatted_date': 'string'
    })

    # Check if master_jobs_applied_to.csv exists
    try:
        master_df = pd.read_csv(file_name)
    except FileNotFoundError:
        master_df = pd.DataFrame(columns=data_df.columns)
    # print(master_df.keys())
    # print(data_df.keys())

    # Check for new entries based on 'job_id'
    new_entries = data_df[~data_df['job_id'].isin(master_df['job_id'])]

    # If there are new entries, append them
    if not new_entries.empty:
        master_df = pd.concat([master_df, new_entries], ignore_index=True)
        master_df.to_csv(file_name, index=False)
    else:
        print("Repeat entry, ignoring.")

def find_show_more_button(driver):
    try:
        return WebDriverWait(driver, 4).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@aria-label="Click to see more description"]'))
        )
    except:
        return None

def find_show_less_button(driver):
    try:
        return WebDriverWait(driver, 4).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@aria-label="Click to see less description"]'))
        )
    except:
        return None
def click_show_more_button(driver):
    toggle_show_button(driver, expand=True)
    
def toggle_show_button(driver, expand):
    """
    Toggle the 'See more' or 'See less' button based on the desired state.

    :param driver: WebDriver instance
    :param expand: True to expand 'See more', False to collapse 'See less'
    """
    if expand:
        # We want to expand the description
        button = find_show_more_button(driver)
        if button:
            perform_click(driver, button, "Show more")
        assert find_show_less_button(driver), "Failed to expand description"
    else:
        # We want to collapse the description
        button = find_show_less_button(driver)
        if button:
            perform_click(driver, button, "Show less")
        assert find_show_more_button(driver), "Failed to collapse description"


def perform_click(driver, button, button_name):
    driver.execute_script("arguments[0].scrollIntoView(true);", button)
    time.sleep(2)  # Slight delay
    try:
        button.click()
    except Exception as e:
        # print(f"Error clicking '{button_name}' button: {e}, attempting JS click")
        driver.execute_script("arguments[0].click();", button)



def get_apply_link(driver):
    try:
        # Open the link in a new tab using JavaScript.
        apply_button = driver.find_element(By.CSS_SELECTOR, ".jobs-apply-button.artdeco-button")
        driver.execute_script("window.open(arguments[0].click(), '_blank');", apply_button)
        # Switch to the new tab.
        driver.switch_to.window(driver.window_handles[1])
        time.sleep(2)
        # Copy the URL.
        apply_link = driver.current_url
        # Close the new tab.
        driver.close()
        # Switch back to the original tab.
        driver.switch_to.window(driver.window_handles[0])
        return apply_link
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_location(element):
    location_div = element.find_element(By.CSS_SELECTOR, ".job-details-jobs-unified-top-card__primary-description div")
    location_parts = location_div.text.split('·')
    potential_location = location_parts[1].strip().split()
    if "," in potential_location[1]:
        return ' '.join(potential_location[:2])
    else:
        return ' '.join(potential_location[:1])

# Define a function to safely extract elements
def safe_extract(element, css_selector, attribute=None):
    try:
        if attribute:
            return element.find_element(By.CSS_SELECTOR, css_selector).get_attribute(attribute)
        return element.find_element(By.CSS_SELECTOR, css_selector).text
    except NoSuchElementException:
        return None

def get_all_skills(driver):
    click_success = click_Show_all_skills_button(driver)
    if not click_success:
        return 'NA', 'NA'
    matched_skills = get_matched_skills(driver)
    unmatched_skills = get_unmatched_skills(driver)
    # click_done_button(driver)
    driver.refresh()
    
    time.sleep(4)
    return matched_skills, unmatched_skills

def click_Show_all_skills_button(driver):

    try:
        # Wait for the button to be present
        button = WebDriverWait(driver, 2).until(
            EC.presence_of_element_located((By.XPATH, '//button/span[text()="Show all skills"]'))
        )

        # Scroll the button into view and click
        driver.execute_script("arguments[0].scrollIntoView(true);", button)
        time.sleep(1)  # sleep can be problematic, use it with caution
        try:
            button.click()
        except:
            driver.execute_script("arguments[0].click();", button)
        time.sleep(1)  # consider using more dynamic waits instead of sleep
    except TimeoutException:
        return False  # Return None if button is not present
    return True


def get_matched_skills(driver):
    try:
        skills_elements = driver.find_elements(By.CSS_SELECTOR, ".job-details-skill-match-status-list__matched-skill div:nth-child(2)")
        return [skill.text for skill in skills_elements]
    except Exception as e:
        print(f"Error in get_matched_skills: {e}")
        return None

def get_unmatched_skills(driver):
    try:
        unmatched_skills_elements = driver.find_elements(By.CSS_SELECTOR, ".job-details-skill-match-status-list__unmatched-skill > div > div[aria-label]")
        return [skill.text for skill in unmatched_skills_elements]
    except Exception as e:
        print(f"Error in get_unmatched_skills: {e}")
        return None
        
def split_and_clean(text):
    return [line.strip() for line in text.split('\n') if line.strip()]


def check_job_id(file_name, job_id):
    try:
        df = pd.read_csv(file_name)
        return job_id in df['job_id'].values
    except FileNotFoundError:
        return False  # File doesn't exist, so job_id isn't there



def save_jobs_data(driver, url2scrape, file_name):
    # Check if url2scrape is a list
    if isinstance(url2scrape, list):
        for url in url2scrape:
            save_jobs_data(driver, url, file_name)
        return

    # Extracting data from the provided HTML
    linked_in_link = url2scrape
    job_id_match = re.search(r'/view/(\d+)/', linked_in_link)
    job_id = int(job_id_match.group(1)) if job_id_match else -1
    # check if it in there first to avoid repeating all the process
    is_repeat = check_job_id(file_name, job_id)
    print(f'JOB ID: {job_id}')
    if is_repeat:
        print('Repeat entry, skipping...')
    else:
        
    
        driver.get(url2scrape)
        
        # driver.refresh()
        time.sleep(6)
        click_show_more_button(driver)
        time.sleep(4)
        
        
    


        
        data = {
            "linked_in_link": linked_in_link,
            "job_id": job_id,
            "job_title": safe_extract(driver, '.job-details-jobs-unified-top-card__job-title'),
            "company_name": safe_extract(driver, '.job-details-jobs-unified-top-card__primary-description a'),
            "apply_link": '',
            "job_location": get_location(driver),
            "time_since_posted": safe_extract(driver, '.tvm__text--neutral span'),
            "num_applicants": safe_extract(driver, '.tvm__text--neutral:last-child'),
            "salary_range": safe_extract(driver, '.job-details-jobs-unified-top-card__job-insight span span:first-child'),
            "job_type": safe_extract(driver, '.ui-label.ui-label--accent-3:nth-child(1) span:first-child'),
            "job_level": safe_extract(driver, '.job-details-jobs-unified-top-card__job-insight-view-model-secondary:nth-child(3)'),
            "company_info": safe_extract(driver, '.job-details-jobs-unified-top-card__job-insight:nth-child(2) span'),
            "description": safe_extract(driver, ".jobs-description-content__text")
        }
    
        
        
        matched_skills, unmatched_skills = get_all_skills(driver)
        
        data['matched_skills'] = matched_skills
        data['unmatched_skills'] = unmatched_skills
        
        date_number = int(datetime.now().strftime('%Y%m%d'))
        
        # Get the formatted date in the format YYYY-MM-DD
        formatted_date = datetime.now().strftime('%Y-%m-%d')
        
        data['date_number'] = date_number
        data['formatted_date'] = formatted_date
        data['apply_link'] = get_apply_link(driver)
        
        
        data = {
            key: [value] for key, value in data.items()
        }
        
        df = pd.DataFrame(data)
        update_csv(df, file_name)
        print('________________________')

