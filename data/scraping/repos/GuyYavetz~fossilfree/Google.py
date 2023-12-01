from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains  
from selenium.common.exceptions import NoSuchElementException
import time
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials as Credentials
import os
import openai

# Set up Google Sheets API credentials
scope = ['https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_json_keyfile_name('client_s.json', scope)

client = gspread.authorize(creds)
sheet = client.open('#USE YOUR SHEET NAME HERE')
worksheet = sheet.get_worksheet(0)
company_names = worksheet.col_values(5)

driver_path = r"#USE YOUR DRIVER PATH HERE"
driver = webdriver.Chrome(executable_path=driver_path)
wait = WebDriverWait(driver, 10)  # wait for up to 10 seconds

try:
    for i in range(542, len(company_names), 10):
        batch_results = []
        for company in company_names[i:i+10]:
            company = company.strip('.')
            driver.get('https://www.google.com')

            search_box = wait.until(EC.element_to_be_clickable((By.NAME, 'q')))
            search_box.send_keys(company)

            actions = ActionChains(driver)
            actions.send_keys(Keys.RETURN)
            actions.perform()

            time.sleep(2)  # wait for the page to load
            try:
                result_stats = driver.find_element(By.ID, 'result-stats').text
                num_results = re.search(r'(\d+(,\d{3})*)', result_stats)
                if num_results:
                    print(f"Company: {company}, Results: {num_results.group(1)}")
                    batch_results.append(num_results.group(1))
            except NoSuchElementException:
                print(f"Company: {company}, Results: Element Not Found")
                continue

        worksheet.update('R' + str(i+2) + ':R' + str(i+2+len(batch_results)-1), [[result] for result in batch_results])
finally:
    driver.quit()
