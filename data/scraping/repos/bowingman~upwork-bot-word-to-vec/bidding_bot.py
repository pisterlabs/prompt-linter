import os
import openai
import random
import time
import datetime
import importlib
import mysql.connector
from gensim.models import word2vec
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sign_in import sign_in_upwork
from sign_up import sign_up_upwork
from create_email import create_new_email
from utils import create_drive, delete_cache, wait_till_page_loaded
from training_bot import get_most_smilar_skills
from selenium.webdriver.common.keys import Keys
from config import skill2prompt

config = importlib.import_module("config")
openai.api_key = "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"


def strip_empty_lines(s):
    lines = s.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)



def generate_proposal(prompt, model="text-davinci-003", temperature=0.8, max_tokens=1500, frequency_penalty=1, presence_penalty=1):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )

    return strip_empty_lines(response["choices"][0]["text"])


def get_skills(element):
    res_skills = []
    try:
        skills_element = element.find_element(
            By.CSS_SELECTOR, "div[class='up-skill-wrapper']").find_elements(By.TAG_NAME, "a")
    except:
        return res_skills
    for skill in skills_element:
        res_skills.append(skill.text)

    return res_skills


def close_dialog(driver):
    try:
        back_svg = driver.find_element(
            By.CSS_SELECTOR, "button[data-test='slider-go-back']")
        back_svg.click()
        wait_till_page_loaded(driver)
    except:
        pass


def start_bidding(driver, email, dataBase, db_cursor):
    model = word2vec.Word2Vec.load("word2vec.model")
    old_job_ids = []
    applied_job_ids = []
    sql_query = (
        "SELECT job_id FROM proposals WHERE full_name = %s AND country = %s ORDER BY id DESC LIMIT 500")
    value = (config.first_name + " " + config.last_name, config.country)
    db_cursor.execute(sql_query, value)
    query_result = db_cursor.fetchall()
    for item in query_result:
        applied_job_ids.append(item[0])
    connects = 50
    is_first = True
    print("Fetched " + str(len(applied_job_ids)) + " applied jobs")

    while connects > 0:
        driver.get(config.url_filtered)
        wait_till_page_loaded(driver)
        time.sleep(5)
        job_lists = driver.find_element(
            By.CSS_SELECTOR, "div[data-test='job-tile-list']").find_elements(By.TAG_NAME, "section")

        for i in range(len(job_lists)):
            # get basic information of the current job
            current_job = job_lists[i]
            current_job_info = {}
            current_job_info['id'] = current_job.get_attribute('id')
            current_job_info['skills'] = get_skills(current_job)
            current_job_info['job_type'] = current_job.find_element(
                By.CSS_SELECTOR, "strong[data-test='job-type']").text
            if current_job_info['id'] in old_job_ids or current_job_info['id'] in applied_job_ids:
                continue
            if current_job_info['job_type'] == 'Fixed-price':
                current_job_info['budget'] = current_job.find_element(
                    By.CSS_SELECTOR, "span[data-test='budget']").text
                current_job_info['job_type'] = "Fixed"
            elif current_job_info['job_type'] == 'Hourly':
                current_job_info['budget'] = None
            elif ':' in current_job_info['job_type']:
                current_job_info['budget'] = current_job_info['job_type'].split(':')[1]
                current_job_info['job_type'] = "Hourly"
            else:
                continue

            time.sleep(5)
            driver.find_element(By.ID, current_job_info['id']).click()
            print("Clicked ", current_job_info['id'])

            # When "We’ve changed how you win work" dialog appears
            if is_first:
                wait_time = 15
                while wait_time > 0:
                    if 'We’ve changed how you win work' in driver.page_source:
                        print("Closed We’ve changed how you win work Modal")
                        actions = ActionChains(driver)
                        actions.move_by_offset(50, 50)
                        actions.click()
                        actions.perform()
                        break
                    time.sleep(1)
                    wait_time = wait_time - 1
            time.sleep(5)

            # When skills are not matched
            try:
                job_detail = driver.find_element(
                    By.CSS_SELECTOR, "div[data-test='job-details-user']")
                current_job_info['title'] = job_detail.find_element(
                    By.TAG_NAME, "h1").text
                current_job_info['description'] = driver.find_element(
                    By.CSS_SELECTOR, "div[data-test='description']").find_element(By.TAG_NAME, "div").text
            except:
                print("Skills are not matched")
                close_dialog(driver)
                continue

            old_job_ids.append(current_job_info['id'])
            time.sleep(2)

            # Click Appy Button
            try:
                apply_btn = driver.find_element(
                    By.CSS_SELECTOR, "button[aria-label='Apply Now']")
                if apply_btn.get_attribute("disabled"):
                    close_dialog(driver)
                    if 'complete your profile' in driver.page_source:
                        connects = 0
                        print("Profile uncompleted")
                        break
                    print("Apply Button is disabled")
                    time.sleep(5)
                    continue
                apply_btn.click()
                print("Clicked Apply Button")
                wait_till_page_loaded(driver)
            except Exception as e:
                connects = 0
                print("Can not click Apply Button")
                close_dialog(driver)
                driver.refresh()
                time.sleep(10)
                continue

            time.sleep(6)
            window_handles = driver.window_handles
            driver.switch_to.window(window_handles[1])
            time.sleep(3)

            # Lack of connects
            try:
                if 'Buy Connects' in driver.page_source:
                    connects = 0
                    print("No connects left")
                    driver.close()
                    window_handles = driver.window_handles
                    driver.switch_to.window(window_handles[0])
                    break
            except:
                pass

            # When the Modal appears for the first time
            if is_first:
                try:
                    modal = driver.find_element(
                        By.CSS_SELECTOR, "div[role='dialog']")
                    if modal:
                        actions = ActionChains(driver)
                        actions.move_by_offset(50, 50)
                        actions.click()
                        actions.perform()
                        time.sleep(5)
                except:
                    pass

            primary_skills = []
            for skill in current_job_info['skills']:
                if skill in model.wv.index_to_key:
                    primary_skills.append(skill)
            additional_skills = get_most_smilar_skills(
                model, current_job_info['skills'], 10)
            if len(additional_skills) > 6:
                skills = primary_skills + random.sample(additional_skills, random.randint(2, 6))
                random.shuffle(skills)
            else:
                skills = []
            print(skills)
            proposal = "The server"
            while proposal.startswith("The server"):
                proposal = generate_proposal(skill2prompt(config.first_name + " " + config.last_name,
                      current_job_info['description'], current_job_info['title'],  skills))
            print(proposal)

            driver.refresh()
            try:
                # Generate Cover Letter
                cover_letter_input = driver.find_element(
                    By.CSS_SELECTOR, "textarea[aria-labelledby='cover_letter_label']")
                driver.execute_script(
                    "arguments[0].scrollIntoView();", cover_letter_input)
                cover_letter_input.send_keys(proposal)
                # Get Questions
                textareas = driver.find_elements(By.TAG_NAME, "textarea")
                print(len(textareas) - 1, " questions exist")
                for i in range(len(textareas)):
                    question = textareas[i].find_element(
                        By.XPATH, ("./../label")).text
                    current_count = 0
                    full_name = config.first_name + " " + config.last_name
                    if question == "Cover Letter":
                        continue
                    sql_query = (
                        "SELECT frequency FROM questions WHERE question = %s and full_name = %s")
                    value = (question, full_name)
                    db_cursor.execute(sql_query, value)
                    query_result = db_cursor.fetchone()
                    if query_result:
                        current_count = int(query_result[0])
                        sql_query = "UPDATE questions SET frequency = %s WHERE question = %s and full_name = %s"
                    else:
                        sql_query = "INSERT INTO questions (frequency, question, full_name) VALUES (%s, %s, %s)"
                    value = (current_count + 1, question, full_name)
                    db_cursor.execute(sql_query, value)
                    dataBase.commit()
                    time.sleep(1)

                if len(textareas) >= 2:
                    now = datetime.datetime.utcnow() - datetime.timedelta(hours=5)
                    sql_query = "INSERT INTO proposals (job_id, full_name, email_address, type, title, description, hourly_budget, skills, country, proposal, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    value = (current_job_info['id'], config.first_name + " " + config.last_name, email, current_job_info['job_type'], current_job_info['title'], current_job_info['description'],
                            current_job_info['budget'], ", ".join(current_job_info['skills']), config.country, "failed, questions, " + proposal, now.strftime('%Y-%m-%d %H:%M:%S'))
                    db_cursor.execute(sql_query, value)
                    dataBase.commit()
            except:
                print("Can not find inputs")
                continue

            send_btn = driver.find_element(
                By.TAG_NAME, "footer").find_element(By.TAG_NAME, "button")
            driver.execute_script("arguments[0].scrollIntoView();", send_btn)
            current_url = driver.current_url
            send_btn.click()
            print("Clicked Send Button")

            # Checkbox Modal
            try:
                modal_element = driver.find_element(
                    By.CSS_SELECTOR, "div[role='dialog']")
                modal_element.find_element(By.CSS_SELECTOR, "div[class='checkbox']").find_element(
                    By.TAG_NAME, "label").click()
                time.sleep(5)
                try:
                    modal_element.find_element(
                        By.XPATH, "//button[contains(text(), 'Submit')]").click()
                except:
                    modal_element.find_element(
                        By.XPATH, "//span[contains(text(), 'Submit')]").click()
                time.sleep(1)
            except:
                pass

            time.sleep(5)
            if "success" in driver.current_url:
                now = datetime.datetime.utcnow() - datetime.timedelta(hours=5)
                sql_query = "INSERT INTO proposals (job_id, full_name, email_address, type, title, description, hourly_budget, skills, country, proposal, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                value = (current_job_info['id'], config.first_name + " " + config.last_name, email, current_job_info['job_type'], current_job_info['title'],
                         current_job_info['description'], current_job_info['budget'], ", ".join(current_job_info['skills']), config.country, proposal, now.strftime('%Y-%m-%d %H:%M:%S'))
                db_cursor.execute(sql_query, value)
                dataBase.commit()
                print("Successfully sent a proposal for ",
                      current_job_info['id'])
            elif current_url == driver.current_url:
                try:
                    send_btn.click()
                except:
                    if "success" in driver.current_url:
                        pass
                    else:
                        try:
                            driver.find_element(By.CSS_SELECTOR, "div[role='dialog']").find_element(
                                By.CSS_SELECTOR, "div[class='checkbox']").find_element(By.TAG_NAME, "label").click()
                            time.sleep(1)
                            try:
                                driver.find_element(By.CSS_SELECTOR, "div[role='dialog']").find_element(
                                    By.XPATH, "//button[contains(text(), 'Continue')]").click()
                                time.sleep(1)
                            except:
                                driver.find_element(By.CSS_SELECTOR, "div[role='dialog']").find_element(
                                    By.XPATH, "//button[contains(text(), 'Submit')]").click()
                                time.sleep(1)
                        except:
                            pass
                        send_btn.click()
                time.sleep(5)
                if current_url != driver.current_url:
                    now = datetime.datetime.utcnow() - datetime.timedelta(hours=5)
                    sql_query = "INSERT INTO proposals (job_id, full_name, email_address, type, title, description, hourly_budget, skills, country, proposal, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    value = (current_job_info['id'], config.first_name + " " + config.last_name, email, current_job_info['job_type'], current_job_info['title'],
                             current_job_info['description'], current_job_info['budget'], ", ".join(current_job_info['skills']), config.country, proposal, now.strftime('%Y-%m-%d %H:%M:%S'))
                    db_cursor.execute(sql_query, value)
                    dataBase.commit()
                    print("2, Successfully sent a proposal for ",
                          current_job_info['id'])
                else:
                    print("33, Failed")

            driver.close()
            window_handles = driver.window_handles
            driver.switch_to.window(window_handles[0])
            close_dialog(driver)
            is_first = False
            driver.get(config.url_filtered)
            time.sleep(10)
            job_lists = driver.find_element(
                    By.CSS_SELECTOR, "div[data-test='job-tile-list']").find_elements(By.TAG_NAME, "section")

        print("Scanned through a page of job lists.")
        time.sleep(3)

    return True, "No connects"


if __name__ == "__main__":
    print("Starting...")
    model = word2vec.Word2Vec.load("word2vec.model")
    dataBase = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="upwork"
    )
    db_cursor = dataBase.cursor()
    current_process = 'Create an email account'
    # current_process = 'Sign in'
    current_process = 'Sign up'
    last_created_email = ""
    status_code = False
    driver = None

    while True:
        print(current_process)
        sql_query = "SELECT email_address FROM accounts WHERE full_name = %s AND country = %s ORDER BY id DESC LIMIT 1"
        value = (config.first_name + " " + config.last_name, config.country)
        db_cursor.execute(sql_query, value)
        query_result = db_cursor.fetchone()
        if query_result is not None:
            last_created_email = query_result[0]

        if current_process == 'Create an email account':
            if driver is None:
                driver = create_drive("localhost:" + str(config.port))
            last_created_email = create_new_email(config.first_name)
            current_process = 'Sign up'
        if current_process == 'Sign up':
            if last_created_email.count("+") > 0:
                current_mail_index = int(
                    query_result[0][query_result[0].index("+")+1:query_result[0].index("@")]) + 1
                if current_mail_index > 9:
                    if driver:
                        driver.quit()
                    time.sleep(3)
                    current_process = "Create an email account"
                    continue
                email = last_created_email.split('+')[0] + f'+{current_mail_index}@' + last_created_email.split('@')[1]
            else:
                current_mail_index = 1
                email = last_created_email.split('@')[0] + f'+{current_mail_index}@' + last_created_email.split('@')[1]
            now = datetime.datetime.utcnow() - datetime.timedelta(hours=5)
            sql_query = "INSERT INTO accounts (full_name, email_address, country, created_at) VALUES (%s, %s, %s, %s)"
            value = (config.first_name + " " + config.last_name, email,
                     config.country, now.strftime('%Y-%m-%d %H:%M:%S'))
            db_cursor.execute(sql_query, value)
            dataBase.commit()
            driver = create_drive("localhost:" + str(config.port))
            delete_cache(driver)
            while current_process == 'Sign up':
                try:
                    status_code, msg = sign_up_upwork(
                        driver, email, dataBase, db_cursor)
                    current_process = 'Send proposals'
                    break
                except Exception as e:
                    print("11", e)
                    continue
        if current_process == "Sign in":
            if driver is None:
                driver = create_drive("localhost:" + str(config.port))
            delete_cache(driver)
            email = last_created_email
            state, msg = sign_in_upwork(driver, email, config.password)
            time.sleep(10)
            current_process = 'Send proposals'
        if current_process == 'Send proposals':
            print("The bidder has started working...")
            if driver is None:
                driver = create_drive("localhost:" + str(config.port))
            while current_process == 'Send proposals':
                try:
                    status_code, msg = start_bidding(
                        driver, email, dataBase, db_cursor)
                    sql_query = "UPDATE accounts SET closed_at = %s WHERE email_address = %s"
                    now = datetime.datetime.utcnow() - datetime.timedelta(hours=5)
                    value = (now.strftime('%Y-%m-%d %H:%M:%S'), email)
                    db_cursor.execute(sql_query, value)
                    dataBase.commit()
                    current_process = 'Sign up'
                    break
                except Exception as e:
                    print("44", e)
                    if driver:
                        driver.quit()
                    driver = create_drive("localhost:" + str(config.port))
                    time.sleep(1)
                    continue
            if driver:
                driver.quit()
            time.sleep(3)
