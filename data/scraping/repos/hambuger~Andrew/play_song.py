from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from openai_util.function_call.openaifunc_decorator import openai_func
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
from airtest.cli.parser import cli_setup
from airtest.core.api import *

executor = ThreadPoolExecutor(10)


@openai_func
def play_song_with_qq_music(song_name: str):
    """
    Use qq music to play songs according to user prompts
    :param song_name: Song name
    """
    executor.submit(play_song, song_name)
    return "playing song：{}".format(song_name)


def play_song(song_name):
    if os.getenv('OS_NAME', 'windows') == 'windows':
        if not cli_setup():
            auto_setup(__file__, logdir=True, devices=["Windows:///"])

        start_app(r"D:\QQMUSIC\QQMusic1942.22.37.01\QQMusic.exe")
        touch(Template(r"tpl1689091307120.png", record_pos=(-0.258, -0.264), resolution=(1920, 1080)))
        text(song_name)
        touch(Template(r"tpl1689095239431.png", record_pos=(-0.191, -0.265), resolution=(1920, 1080)))
        touch(Template(r"tpl1689095337138.png", record_pos=(-0.35, -0.142), resolution=(1920, 1080)))
        touch(Template(r"tpl1689095393492.png", record_pos=(0.451, -0.265), resolution=(1920, 1080)))

    else:
        options = webdriver.ChromeOptions()
        options.add_argument(r"user-data-dir=D:\Python\Tools\chromedriver_win32\data")
        options.add_argument("--start-maximized")  # maximize window
        options.add_argument("--no-sandbox")  # disable sandbox
        options.add_argument("--disable-dev-shm-usage")  # Disable developer mode
        driver_path = r'D:\Python\Tools\chromedriver_win32\chromedriver'
        webdriver_service = Service(driver_path)
        driver = webdriver.Chrome(service=webdriver_service, options=options)
        # Navigate to QQ Music
        driver.get("https:/y.qq.com/")
        # Enter the song name in the search box
        search_field = driver.find_element(By.CLASS_NAME, "search_input__input")
        search_field.clear()
        # Enter and submit a search request
        search_field.send_keys(song_name)
        # Find and hit the play button
        play_button = driver.find_element(By.CLASS_NAME, "search_input__btn")
        play_button.click()
        time.sleep(3)  # Pause for 3 seconds
        play_button = driver.find_element(By.LINK_TEXT, "播放全部")
        play_button.click()
        time.sleep(3)  # Pause for 3 seconds
        element = driver.find_elements(By.CLASS_NAME, "songlist__time")
        driver.minimize_window()
        time_str = element[0].text
        # split string and convert to seconds
        minutes, seconds = map(int, time_str.split(":"))
        total_seconds = minutes * 60 + seconds
        time.sleep(total_seconds)
        driver.quit()

# play_song("Welcome Home, Son")
