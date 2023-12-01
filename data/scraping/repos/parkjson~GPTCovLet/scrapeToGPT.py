import openai
import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# pytest main.py --cov

"""
scrapeToGPT.py
====================================
The core module of my cover letter generation project
"""


def urlScrape(url):
    """
    Boots up a chrome browser window and logs in. With your credentials.

    Parameters
    ----------
    url
        Your typical www.linkin.com/... url of a typical job post. It needs your linkedin username and password.
    """
    options = Options()
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

    driver.get("https://www.linkedin.com/login")
    time.sleep(2)
    username = driver.find_element(By.ID, "username")
    username.send_keys("")
    pword = driver.find_element(By.ID, "password")
    pword.send_keys("")
    driver.find_element(By.XPATH, "//button[@type='submit']").click()

    driver.get(url)
    time.sleep(3)
    src = driver.page_source
    html = BeautifulSoup(src, "html.parser")

    return descScrape(html)


def descScrape(html):
    """
    Webscrapes the html description of the LinkedIn url of a job posting.

    Parameters
    ----------
    html
        The html scraped from the urlScrape function automatically goes here.
    """
    # print(html.prettify())
    company_name_html = html.find_all("a", {"class": "ember-view t-black t-normal"})
    company_name = (
        company_name_html[0].text
    ).strip()  # if there is an error here it means you need to input your linkedin user email and password in the urlScrape Function

    company_desc_html = html.find_all("div", {"class": "jobs-description"})

    company_desc = (company_desc_html[0].text).strip()
    # print(company_desc)

    info = company_name + ". Here is the relevant job description. " + company_desc
    return info


# def pull templateCoverLetter or Prompt():


def completionQuery(desc):
    """
    Takes the description and combines it with a preset query to send to openAI.

    Parameters
    ----------
    desc
        Description from the html of descScrape is automatically put in here. You must also enter your openAI api key.
    """
    openai.api_key = ""

    # pull templateCoverLetterHere
    # cap completion to 10tokens
    prompt = (
        "Write a genuine and human three paragraph cover letter for the position of software developer to the company "
        + desc
        + ". I have an interest in the company's mission, which you should explicitly find out. Align with key facts about me below. I'm a recent graduate of Columbia University who studied computer science. Additional key facts to include are: 1: I have experience in open source development, both maintaining and contributing to GitHub projects. This has brought me up to the industry's best practices. 2: My previous internship in a startup has trained me to learn and adapt quickly. 3: During my personal project in cofounding a logistics centralization app for my university, I have learned to work alongside colleagues, both technical and laypersons. Sign off with the name \"Jaesung Park\"."
    )
    print("Prompt:")
    print(prompt)
    completion = openai.Completion.create(
        model="text-davinci-003", prompt=prompt, max_tokens=1500, temperature=0.6
    )
    print(completion.choices[0].text)
    return True


url = "https://www.linkedin.com/jobs/view/jr-mid-level-software-engineer-roku-remote-at-tandym-group-3555277192/?utm_campaign=google_jobs_apply&utm_source=google_jobs_apply&utm_medium=organic"
# completionQuery(urlScrape(url))
# print(urlScrape(url))
