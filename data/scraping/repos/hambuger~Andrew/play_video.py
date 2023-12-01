import json
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from youtube_search import YoutubeSearch
from openai_util.function_call.openaifunc_decorator import openai_func
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(10)


@openai_func
def play_video(query: str):
    """
    Play the appropriate video based on user input
    :param query:user query
    :return:
    """
    executor.submit(inner_play, query)
    return "playing video：{}".format(query)


def inner_play(query):
    results = YoutubeSearch(query, 1).to_json()
    data = json.loads(results)
    url_suffix_list = [video["url_suffix"] for video in data["videos"]]
    options = webdriver.ChromeOptions()
    options.add_argument(r"user-data-dir=D:\Python\Tools\chromedriver_win32\data")
    options.add_argument("--start-maximized")  # maximize window
    options.add_argument("--no-sandbox")  # disable sandbox
    options.add_argument("--disable-dev-shm-usage")  # Disable developer mode
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    driver_path = r'D:\Python\Tools\chromedriver_win32\chromedriver'
    webdriver_service = Service(driver_path)
    driver = webdriver.Chrome(service=webdriver_service, options=options)
    # open play page
    driver.get("https://youtube.com" + url_suffix_list[0])
    # Navigate to the body element, then send the key "f"
    time.sleep(2)
    body = driver.find_element(By.TAG_NAME, "body")
    body.send_keys('f')

    while True:
        try:
            # Get the state of the player
            player_status = driver.find_element(By.CSS_SELECTOR, ".html5-video-player").get_attribute("data-state")
            # If the player's status is "End", then the video has finished playing
            if player_status == "ended":
                driver.quit()
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break
        # check every once in a while
        time.sleep(5)
