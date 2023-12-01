from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from openai_utils import ask
import time
import json
import sqlite3
import re


class JobListing:
    def __init__(self, title, company, location, link, easy_apply, remote):
        self.title = title
        self.company = company
        self.location = location
        self.link = link
        self.easy_apply = easy_apply
        self.remote = remote
        self.description = ""

    def set_description(self, description):
        self.description = description

    def get_easy_apply_str(self):
        if self.easy_apply:
            return "Easy Apply"
        else:
            return "Not Easy Apply"

    def __str__(self):
        return (
            self.title
            + ", "
            + self.company
            + ", "
            + self.location
            + ", "
            + self.link
            + ", "
            + self.description
            + ", "
            + self.get_easy_apply_str()
            + ", "
            + self.remote
        )


def process_config():
    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)
    return config


def launch_driver(url):
    service = Service()
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    return driver


def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Exception as e:
        print(e)
    return None


def login(driver, username, password):
    email_field = 'session_key'
    password_field = 'session_password'
    sign_in_button = '//*[@id="main-content"]/section[1]/div/div/form/div[2]/button'

    driver.find_element(By.ID, email_field).send_keys(
        username)
    driver.find_element(By.ID, password_field).send_keys(
        password)
    driver.find_element(
        By.XPATH, sign_in_button).click()
    driver.get("https://www.linkedin.com/jobs/")


def search(driver, job, location):
    driver.set_window_size(1080, 1080)
    driver.get("https://www.linkedin.com/jobs")
    time.sleep(2)
    try:
        driver.find_element(
            By.XPATH, '//*[contains(@id, "jobs-search-box-keyword-id")]').clear()  # clear job title
        driver.find_element(
            By.XPATH, '//*[contains(@id, "jobs-search-box-location-id")]').clear()  # clear location
    except:
        pass
    driver.find_element(
        By.XPATH, '//*[contains(@id, "jobs-search-box-keyword-id")]').send_keys(job)  # fill job title
    time.sleep(2)
    try:
        driver.find_element(
            By.XPATH, '//*[contains(@id, "jobs-search-box-location-id")]').send_keys(location)  # fill location
        driver.find_element(
            By.XPATH, '//*[@id="global-nav-search"]/div/div[2]/button[1]').click()  # search
    except:
        # press enter
        driver.find_element(
            By.XPATH, '//*[contains(@id, "jobs-search-box-keyword-id")]').send_keys(Keys.ENTER)  # fill job title
        pass
    time.sleep(4)
    return driver.current_url


def get_job_listings(driver):
    listings = []
    driver.set_window_size(360, 640)
    scrollHeight = 0

    try:
        driver.find_element(
            By.CLASS_NAME, "scaffold-layout__detail-back-button").click()
    except:
        pass
    time.sleep(2)
    for i in range(0, 24):
        soup = BeautifulSoup(driver.page_source, "html.parser")
        listing = soup.find_all(
            "li", {"class": "jobs-search-results__list-item"})[i]

        # scroll
        scrollHeight += 135.969
        driver.execute_script("window.scrollTo(0, " + str(scrollHeight) + ");")
        time.sleep(1)

        title = listing.find(
            "a", {"class": "job-card-list__title"}).text.strip()
        company = listing.find(
            class_="job-card-container__primary-description").text.strip()
        location = listing.find(
            "li", {"class": "job-card-container__metadata-item"}).text.strip()
        link = listing.find(
            "a", {"class": "job-card-list__title"})["href"].strip()
        try:
            _easy_apply = listing.find(
                "li", {"class": "job-card-container__apply-method"}).text.strip()
            easy_apply = ("Easy Apply" in _easy_apply)
        except:
            easy_apply = False
        try:
            remote = "Remote" if ("Remote" in location) else "Not Remote"
        except:
            remote = "Not Remote"
        listings.append(JobListing(
            title, company, location, link, easy_apply, remote))

    return listings


def apply(listing: JobListing, driver, db, connection):
    try:
        apply_to_listing(driver, listing)

        # next button loop
        count = 0
        while True:
            count += 1
            if (count > 10):
                break

            else:
                try:
                    driver.find_element(
                        By.XPATH, "//button[@aria-label='Continue to next step']").click()
                    driver.find_element(
                        By.XPATH, "//button[@aria-label='Continue to next step']").click()
                except:
                    break

                # # get form fields
                form = driver.find_element(
                    By.CLASS_NAME, "jobs-easy-apply-modal")
                form_groups = form.find_elements(
                    By.CLASS_NAME, "jobs-easy-apply-form-section__grouping")

                for group in form_groups:
                    label = group.find_element(By.TAG_NAME, "label")
                    select = input = textarea = None
                    try:
                        input = group.find_element(By.TAG_NAME, "input")
                        if 'year' in label.text.lower() or 'how long' in label.text.lower():
                            input.clear()
                            input.send_keys("5")
                            continue
                        elif input.get_attribute("type") == "radio":
                            input.click()
                            continue
                        continue
                    except:
                        pass

                    try:
                        textarea = group.find_element(By.TAG_NAME, "textarea")
                        response = ask(label.text, listing.description, 4)
                        textarea.send_keys(response)
                        continue
                    except:
                        pass

                    try:
                        select = group.find_element(By.TAG_NAME, "select")
                        select_label = select.accessible_name
                        if "english" in select_label.lower():
                            continue
                        options = select.find_elements(By.TAG_NAME, "option")
                        options[1].click()
                        continue
                    except:
                        pass

                try:
                    driver.find_element(
                        By.XPATH, "//button[@aria-label='Review your application']").click()
                    time.sleep(1)
                except Exception as e:
                    print(e)

        try:
            driver.find_element(
                By.XPATH, "//button[@aria-label='Review your application']").click()
            time.sleep(1)
        except Exception as e:
            print(e)

        try:
            driver.find_element(
                By.XPATH, "//button[@aria-label='Submit application']").click()

            db.execute("INSERT INTO listings (title, company, location, link, description, easy_apply, remote) VALUES (?, ?, ?, ?, ?, ?, ?)", (
                listing.title, listing.company, listing.location, listing.link, listing.description, listing.easy_apply, listing.remote))
            connection.commit()
            time.sleep(1)
        except Exception as e:
            print(e)
    except:
        pass


def apply_to_listing(driver, listing):
    driver.get("https://www.linkedin.com" + listing.link)
    time.sleep(2)
    driver.set_window_size(1080, 920)
    description = get_description(driver, listing)
    listing.set_description(description)
    try:
        # scroll to top of page
        driver.execute_script("window.scrollTo(0, 0);")
        driver.find_element(By.CLASS_NAME, "jobs-apply-button").click()
    except NoSuchElementException:
        driver.back()
        return


def get_description(driver, listing):
    driver.set_window_size(1080, 920)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    # scroll down one page
    driver.execute_script("window.scrollTo(0, 1080);")
    see_more = driver.find_element(
        By.XPATH, "//button[@aria-label='Click to see more description']")
    driver.execute_script("arguments[0].click();", see_more)
    description = soup.find(
        "div", {"id": "job-details"}).text.strip()
    return description


def next_page(driver, i, search_url):
    driver.set_window_size(1080, 920)
    # remove currentJobId parameter from url
    # url = search_url.split("currentJobId")[0]
    # https://www.linkedin.com/jobs/search/?currentJobId=3155175274&f_WT=2&keywords=Full%20Stack&refresh=true
    # remove currentJobId parameter from url using regex
    url = re.sub(r'&currentJobId=\d+', '', search_url)
    url = search_url + "&start=" + str(i*25)
    driver.get(url)


if __name__ == "__main__":

    driver = launch_driver("https://www.linkedin.com/")
    config = process_config()
    username = config["username"]
    password = config["password"]
    job_titles = config["job_titles"]
    locations = config["locations"]
    connection = create_connection("jobs.db")
    db = connection.cursor()
    db.execute("CREATE TABLE IF NOT EXISTS listings (id INTEGER PRIMARY KEY, title TEXT, company TEXT, location TEXT, link TEXT, description TEXT, easy_apply TEXT, remote TEXT)")
    connection.commit()
    pages = 40
    time.sleep(10)

    login(driver, username, password)

    for job in job_titles:
        for location in locations:
            time.sleep(2)
            search_url = search(driver, job, location)
            time.sleep(2)
            for i in range(1, pages + 1, 1):
                try:
                    listings = get_job_listings(driver)
                    listings = [x for x in listings if x.easy_apply]
                except:
                    listings = []
                time.sleep(5)
                for listing in listings:
                    time.sleep(5)
                    apply(listing, driver, db, connection)

                next_page(driver, i, search_url)
                time.sleep(5)
