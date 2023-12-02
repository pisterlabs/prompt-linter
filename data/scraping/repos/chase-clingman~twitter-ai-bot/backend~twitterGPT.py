import time
import json
import random

import undetected_chromedriver as uc
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
# nosuchelementexception
# from selenium.common.exceptions import NoSuchElementException

import os
import openai

# troubleshooting
from bs4 import BeautifulSoup

from PIL import Image
from io import BytesIO

openai.api_key = 'sk-'

# GMAIL = 'zaitsev.jidkov@gmail.com'
# PASSWORD = '94MhLIBh0K'
PERSONALITIES = [
    {"style": "formal", "role": "business analyst"},
    {"style": "casual", "role": "regular customer"},
    {"style": "humorous", "role": "comedy writer"},
    {"style": "detailed", "role": "food critic"},
    {"style": "concise", "role": "busy parent"},
    {"style": "concise", "role": "time-pressed executive"},
    {"style": "concise", "role": "college student"},
    {"style": "concise", "role": "busy mom"},
    {"style": "concise", "role": "fitness enthusiast"},
    {"style": "concise", "role": "world traveler"},
    {"style": "concise", "role": "retiree"},
    {"style": "concise", "role": "freelance writer"},
    {"style": "concise", "role": "tech worker"},
    {"style": "concise", "role": "film critic"},
    {"style": "concise", "role": "restaurant owner"},
    {"style": "concise", "role": "local guide"},
    {"style": "concise", "role": "musician"},
    {"style": "concise", "role": "architect"},
    {"style": "concise", "role": "teacher"},
    {"style": "concise", "role": "doctor"},
    {"style": "concise", "role": "engineer"},
    {"style": "concise", "role": "sports fan"},
    {"style": "concise", "role": "interior designer"},
    {"style": "concise", "role": "gardener"},
    {"style": "concise", "role": "art critic"},
    # add as many personalities as you like
]
ACCOUNTS = [

    # add as many accounts as you like
]


def login_to_google_account(driver, account):
    email = account["email"]
    password = account["password"]
    recovery = account["recovery"]

    driver.get(
        "https://twitter.com/i/flow/login")
    time.sleep(15)
    # Explicit wait for the button to be clickable
    try:
        # make action chain that tabs and then presses enter
        actions = ActionChains(driver)
        actions.send_keys(Keys.TAB)
        actions.send_keys(Keys.ENTER)
        actions.perform()
    except Exception as e:
        print(e)
    time.sleep(2)
    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(2)

    driver.find_element(
        By.XPATH, "/html/body/div[1]/div[1]/div[2]/div/c-wiz/div/div[2]/div/div[1]/div/form/span/section/div/div/div[1]/div/div[1]/div/div[1]/input").send_keys(email)
    driver.find_element(
        By.XPATH, "/html/body/div[1]/div[1]/div[2]/div/c-wiz/div/div[2]/div/div[2]/div/div[1]/div/div/button").click()

    time.sleep(5)

    driver.find_element(
        By.XPATH, "/html/body/div[1]/div[1]/div[2]/div/c-wiz/div/div[2]/div/div[1]/div/form/span/section[2]/div/div/div[1]/div[1]/div/div/div/div/div[1]/div/div[1]/input").send_keys(password)
    driver.find_element(
        By.XPATH, "/html/body/div[1]/div[1]/div[2]/div/c-wiz/div/div[2]/div/div[2]/div/div[1]/div/div/button").click()

    time.sleep(5)
    # screnshot 1
    # full_screenshot(driver, "1")
    try:
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
            (By.XPATH, "/html/body/div/c-wiz/div/div/div/div[2]/div[4]/div[1]/button"))).click()
    except:
        print("The element was not found.")

    time.sleep(5)
    # CONFIRM TO ALLOW TWITTER
    try:
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
            (By.XPATH, '//*[@id="confirm_yes"]'))).click()
        time.sleep(5)
        driver.switch_to.window(driver.window_handles[-1])
        # make an action chain for typing 'j' and then tab and then '1' and then tab and then '1990' and then tab and then 'enter'
        actions = ActionChains(driver)
        actions.send_keys("j")
        actions.send_keys(Keys.TAB)
        actions.send_keys("1")
        actions.send_keys(Keys.TAB)
        actions.send_keys("1990")
        actions.send_keys(Keys.TAB)
        actions.send_keys(Keys.ENTER)
        actions.perform()

        time.sleep(5)

        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
            (By.XPATH, '//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div/div/div'))).click()
        time.sleep(5)

        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
            (By.XPATH, '//*[@id="layers"]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div/div/div'))).click()
        time.sleep(5)

        driver.switch_to.window(driver.window_handles[-1])
    except:
        print("The element was not found.")
    try:
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
            (By.XPATH, "/html/body/div/div[1]/div/div/main/div[4]/div[1]"))).click()
    except:
        print("The second element was not found.")

    try:
        driver.find_element(
            By.XPATH, "/html/body/div[1]/div[1]/div[2]/div/div[2]/div/div/div[2]/div/div[1]/div/form/span/section/div/div/div/ul/li[3]/div").click()
        time.sleep(10)
        # full_screenshot(driver, "2")
        try:

            driver.find_element(
                By.XPATH, "/html/body/div[1]/div[1]/div[2]/div/div[2]/div/div/div[2]/div/div[1]/div/form/span/section/div/div/div[2]/div[1]/div/div[1]/div/div[1]/input").send_keys(recovery)

        except:

            print("The recovery input was not found.")
        # full_screenshot(driver, "3")
        driver.find_element(
            By.XPATH, "/html/body/div[1]/div[1]/div[2]/div/div[2]/div/div/div[2]/div/div[2]/div/div[1]/div/div/button").click()

    except:
        print("The third element was not found.")

    time.sleep(5)
    print('Logged in to Google account.')


# def find_account(driver, account):
#     time.sleep(2)
#     driver.switch_to.window(driver.window_handles[-1])
#     time.sleep(2)
#     driver.get(account)
#     time.sleep(2)
#     print('Found account.')
#     # driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#     time.sleep(4)
#     # scroll back to top
#     # driver.execute_script("window.scrollTo(0, 0);")
#     # time.sleep(2)

#     # Wait until the div with the specified aria-label is present
#     WebDriverWait(driver, 10).until(EC.presence_of_element_located(
#         (By.XPATH, '//div[contains(@aria-label, "Timeline: ")]')))

#     # Then, find that div
#     timeline_div = driver.find_element(
#         By.XPATH, '//div[contains(@aria-label, "Timeline: ")]')

#     articles = timeline_div.find_elements(
#         By.CSS_SELECTOR, ".css-1dbjc4n.r-1loqt21.r-18u37iz.r-1ny4l3l.r-1udh08x.r-1qhn6m8.r-i023vh.r-o7ynqc.r-6416eg")

#     # remove the first article, which is a pinned tweet
#     articles.pop(0)

#     for article in articles:

#         article_text = article.text
#         time.sleep(2)
#         # get article in view
#         time.sleep(2)
#         article.click()  # click on the article
#         time.sleep(2)  # wait for the article to load

#         # Extract text from the article
#         # replace 'selector-for-article-text' with the actual CSS selector

#         # print(article_text)

#         # Now you can pass 'article_text' to OpenAI API. For example:
#         # random_personality = random.choice(PERSONALITIES)
#         review_instruction = f"Reply to the tweet that the user provides in an extremely rude way. Max 280 characters."

#         response = openai.ChatCompletion.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": review_instruction},
#                 {"role": "user", "content": article_text}
#             ]
#         )

#         reply = response['choices'][0]['message']['content']
#         print(reply)

#         try:
#             WebDriverWait(driver, 10).until(EC.presence_of_element_located(
#                 (By.XPATH, '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/section/div/div/div/div/div[1]/div/div/div/div/div[2]/div[1]/div/div/div/div[2]/div[1]/div/div/div/div/div[1]/div/div/div/div/div/label/div[1]/div/div/div/div/div/div[2]/div')))
#             reply_box = driver.find_element(
#                 By.XPATH, '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/section/div/div/div/div/div[1]/div/div/div/div/div[2]/div[1]/div/div/div/div[2]/div[1]/div/div/div/div/div[1]/div/div/div/div/div/label/div[1]/div/div/div/div/div/div[2]/div')
#             reply_box.click()
#             reply_box.send_keys(reply)
#             time.sleep(2)
#             # make a action chain of 4 tabs, and then enter
#             ActionChains(driver).send_keys(
#                 Keys.TAB * 4).send_keys(Keys.ENTER).perform()
#             time.sleep(15)
#         except:
#             print("Reply box not found.")

#         driver.back()
#         time.sleep(15)

#     driver.quit()


def find_tweet(driver, tweet_url):
    time.sleep(2)
    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(2)
    driver.get(tweet_url)
    time.sleep(15)
    print('Found tweet.')

    tweet_text = driver.find_element(
        By.XPATH, '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/section/div/div/div/div/div[1]/div/div/article').text

    review_instruction = f"Reply to the tweet that the user provides in an extremely rude way. Max 250 characters."

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": review_instruction},
            {"role": "user", "content": tweet_text}
        ]
    )

    reply = response['choices'][0]['message']['content']
    print(reply)

    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located(
            (By.XPATH, '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/section/div/div/div/div/div[1]/div/div/div/div/div[2]/div[1]/div/div/div/div[2]/div[1]/div/div/div/div/div[1]/div/div/div/div/div/label/div[1]/div/div/div/div/div/div[2]/div')))
        reply_box = driver.find_element(
            By.XPATH, '//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]/div/section/div/div/div/div/div[1]/div/div/div/div/div[2]/div[1]/div/div/div/div[2]/div[1]/div/div/div/div/div[1]/div/div/div/div/div/label/div[1]/div/div/div/div/div/div[2]/div')
        reply_box.click()
        reply_box.send_keys(reply)
        time.sleep(2)
        # make a action chain of 4 tabs, and then enter
        ActionChains(driver).send_keys(
            Keys.TAB * 4).send_keys(Keys.ENTER).perform()
        time.sleep(15)
    except:
        print("Reply box not found.")


chrome_options = uc.ChromeOptions()
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-popup-blocking")
chrome_options.add_argument("--profile-directory=Default")
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--disable-plugins-discovery")
chrome_options.add_argument("--incognito")
# chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--window-size=1920x1080")
chrome_options.add_argument(
    "user_agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")

driver = uc.Chrome(options=chrome_options)
driver.delete_all_cookies()

account = random.choice(ACCOUNTS)
login_to_google_account(driver, account)

# find_account(driver, "https://twitter.com/CannonCJohnson")
find_tweet(driver, "https://twitter.com/JakeSucky/status/1673817187302404096")
driver.quit()
