"""
Here the core of the Bot,
all the mehtods to handle commands.
"""


import logging
import os
from time import sleep
from typing import List

import openai
import openai.error
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as ec


os.system("rm -rf profile")  # REMOVE OLD PROFILE

logger = logging.getLogger(__name__)
OPENAI_TOKEN = os.environ.get("OPENAI_TOKEN", None)
DRIVER_PATH = "chromedriver"
WHATSAPP_WEB = "https://web.whatsapp.com/"

# You shouldn't need to change this
CLASSES = {
    "msg_text": "_21Ahp",
    "msg_box": "_3Uu1_",
    "send_button": "epia9gcq",
    "search_box": "_2vDPL",
    "loading": "_2dfCc",
}

XPATH = {
    "left_msg": "span.selectable-text",
    "media_button": "span[data-icon='clip']",
    "media_input": "input[type='file']",
    "send_button": "//div[contains(@class, 'iA40b')]",
}


class Bot:
    """The bot implementaion
    this class creates the webdriver and manages it.
    It establishes connection with OpenAI API.
    """

    dir_path = os.getcwd()
    chromedriver = DRIVER_PATH
    profile = os.path.join(dir_path, "profile", "wpp")

    def __init__(self):
        if OPENAI_TOKEN:
            openai.api_key = OPENAI_TOKEN
            self.conversations = []
        self.messages = []
        self.options = webdriver.ChromeOptions()
        self.options.add_argument(r"user-data-dir={}".format(self.profile))
        # start webdriver
        self.driver = webdriver.Chrome(self.chromedriver, chrome_options=self.options)
        self.driver.get(WHATSAPP_WEB)
        self.driver.implicitly_wait(3)

    def get_last_message(self) -> str:
        try:
            # get the last message
            self.messages = self.get_all_visible_messages()
            self.msg_element = self.messages[-1]
            return self.get_message_text(self.msg_element)

        except Exception as e:
            logger.error(f"Error getting message: {e}")
            return ""

    def reply(self, message):
        try:
            # hover
            el = self.msg_element.find_element_by_class_name("_3mSPV")
            ActionChains(self.driver).move_to_element(el).perform()
            # click options
            self.driver.find_element_by_xpath(
                '//div[@aria-label="Context Menu"]'
            ).click()
            WebDriverWait(self.driver, 5).until(
                ec.visibility_of_element_located((By.CLASS_NAME, "_1MZM5"))
            )
            sleep(0.5)
            # choose reply
            for choice in self.driver.find_elements_by_class_name("_1MZM5"):
                if choice.text == "Reply":
                    choice.click()
                    sleep(0.1)
                    break
            logger.info("A message is selected.")
            self.send_message(message)
        except TimeoutException as e:
            logger.error(e)
        except Exception as e:
            logger.error(f"Erorr: in `reply`: {e}")

    def send_message(self, msg):
        try:
            sleep(0.1)
            # select box message and typing
            self.msg_box_element = self.driver.find_element_by_class_name(
                CLASSES["msg_box"]
            )
            for part in msg.split("\n"):
                self.msg_box_element.send_keys(part)
                ActionChains(self.driver).key_down(Keys.SHIFT).key_down(
                    Keys.ENTER
                ).key_up(Keys.SHIFT).key_up(Keys.ENTER).perform()
            sleep(0.5)
            # Select send button
            self.send_btn_element = self.driver.find_element_by_class_name(
                CLASSES["send_button"]
            )
            # click
            self.send_btn_element.click()
            sleep(0.5)
        except Exception as e:
            logger.error(f"Error send message: {e}")

    def send_media(self, file):
        try:
            self.driver.find_element_by_css_selector(XPATH["media_button"]).click()
            attach = self.driver.find_element_by_css_selector(XPATH["media_input"])
            attach.send_keys(fileToSend)
            sleep(3)
            send = self.driver.find_element_by_xpath(XPATH["send_button"])
            send.click()
        except Exception as e:
            print("Error send media", e)

    def is_ready(self):
        """By checking if the seach box is availble
        we know it's ready to chat"""

        return bool(
            self.driver.find_elements_by_xpath(
                "//div[@data-testid='chat-list-search-container']"
            )
        )

    def set_chat(self, chat_name):
        """Open specific chat"""

        while not self.is_ready():
            sleep(1)

        sleep(5)
        try:
            self.search_chat_element = self.driver.find_element_by_xpath(
                "//div[@data-testid='chat-list-search']"
            )
            self.search_chat_element.send_keys(chat_name)
            sleep(2)
            self.chat_element = self.driver.find_element_by_xpath(
                f"//span[contains(@title, '{chat_name}')]"
            )
            self.chat_element.click()
            self.search_chat_element.clear()
        except Exception as e:
            logger.warn(f"Issue in setting chat {chat_name}, {e}")

    def loop(self, handle):

        while True:
            if self.is_ready():
                msg = self.get_last_message()
                if msg:
                    handle(msg)

            sleep(1)

    def ask_gpt(self, msg, max_tokens=50):

        logger.info(f"Prompe: {msg}")
        if not OPENAI_TOKEN:
            print("Make sure you setup OpenAI token.")

        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=msg,
                temperature=0.3,
                max_tokens=max_tokens,
            )

        except openai.InvalidRequestError as e:
            logger.error(e)
            self.reply(e)
            return

        except openai.APIError as e:
            logger.error(e)
            self.send_message("APIConnectionError, try again latter")
            return

        except openai.OpenAIError as e:
            logger.error(e)
            self.reply("A fatal error ocurred")
            return

        text = self._clean_text(response.to_dict()["choices"][0].to_dict()["text"])
        self.reply(f"> {text}")

    def construct_conversation(self, prompt: str):
        """This is to help constructing
        the `messages` parameter for `ChstCompletion`
        so it can biuld an answer based on prevoius responses"""

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for students of King Fahd University of Petroleum and Minerals (KFUPM).",
            },
        ]
        messages.extend(self.conversations)
        messages.append({"role": "user", "content": prompt})

        return messages

    def store_response(self, completion):
        """To store new responses as 'assistant'"""

        self.conversations.append(completion.choices[0]["message"].to_dict())

        # to prevent excceding the max tokens:
        # 4096 tokens for gpt-3.5-turbo-0301
        if completion.usage.total_tokens > 2000:
            self.conversations.pop(0)

    @staticmethod
    def _clean_text(text: str) -> str:
        return text.replace("\t", " " * 4).replace("\b", "")

    def ask_chat_gpt(self, msg, new_thread=False):

        logger.info(f"Prompe: {msg}")
        if not OPENAI_TOKEN:
            print("Make sure you setup OpenAI token.")

        try:
            if new_thread:

                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=str(msg),
                )
            else:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=self.construct_conversation(msg),
                )

        except openai.InvalidRequestError as e:
            logger.error(e)
            self.reply(e)
            return

        except openai.APIError as e:
            logger.error(e)
            self.send_message("APIConnectionError, try again latter")
            return

        except openai.OpenAIError as e:
            logger.error(e)
            self.reply("A fatal error ocurred")
            return

        self.store_response(completion)
        cleaned = self._clean_text(completion.choices[0].message.content)
        self.reply(f"> {cleaned}")

    def ask_DALL_E(self, msg):

        logger.info(f"Prompe: {msg}")
        try:
            response = openai.Image.create(prompt=msg, n=1, size="512x512")

        except openai.InvalidRequestError as e:
            logger.error(e)
            self.reply(str(e))
            return

        except openai.APIError as e:
            logger.error(e)
            self.send_message("APIConnectionError, try again latter")
            return

        except openai.OpenAIError as e:
            logger.error(e)
            self.reply("A fatal error ocurred")
            return

        image_url = response["data"][0]["url"]

        self.reply(image_url)

    def is_reply(self, msg_box: WebElement) -> bool:
        """Wehther this message is a reply to another one."""
        return bool(msg_box.find_elements_by_class_name("_1hl2r"))

    def get_all_visible_messages(self) -> List[WebElement]:
        return self.driver.find_element_by_class_name(
            "_2Ex_b"
        ).find_elements_by_class_name("_7GVCb")

    def get_replied_text(self, msg_box: WebElement) -> str:
        try:
            return msg_box.find_element_by_class_name("_37DQv").text
        except Exception as e:
            print(e)

    def go_to_replied_message(self, msg_box: WebElement) -> WebElement | None:
        try:
            msg_box.find_element_by_class_name("_37DQv").click()
            detected = self.driver.find_elements_by_class_name("velocity-animating")
            for el in detected:
                try:
                    if "_7GVCb" in el.get_attribute("class"):
                        return el
                except:
                    pass

        except Exception as e:
            print(f"I couldn't find the replied message: {e}")

    @property
    def is_sender_me(self) -> bool:
        return "message-out" in self.msg_element.get_attribute("class")

    def get_message_text(self, msg_box: WebElement) -> str:
        return msg_box.find_element_by_class_name("_21Ahp").text

    def get_contact(self, msg_box) -> str:
        return msg_box.find_element_by_class_name("_3FuDI").text

    def find_message_element(self, context: str, messages) -> WebElement | None:
        for el in reversed(messages):
            try:
                if context in self.get_message_text(el):
                    return el
            except:
                pass

    def remove_participant(self, name: str) -> bool:
        # chat option
        # bot.driver.find_element_by_class_name('kiiy14zj')
        WebDriverWait(self.driver, 10).until(
            ec.visibility_of_element_located((By.CLASS_NAME, "kiiy14zj"))
        ).click()

        # choose an option
        WebDriverWait(self.driver, 10).until(
            ec.visibility_of_all_elements_located((By.CLASS_NAME, "_1MZM5"))
        )[0].click()

        # Click search icon
        options = WebDriverWait(self.driver, 10).until(
            ec.visibility_of_all_elements_located(
                (By.XPATH, '//span[@data-icon="search"]')
            )
        )
        options[1].click()
        # write
        self.driver.find_element_by_xpath(
            '//div[@data-testid="chat-list-search"]'
        ).send_keys(name)
        sleep(1)
        # search for the box el
        els = self.driver.find_element_by_xpath(
            '//div[@data-testid="popup-contents"]'
        ).find_elements_by_tag_name("span")
        for el in els:
            if name in el.text:
                el.click()
        # Click remove
        options = WebDriverWait(self.driver, 10).until(
            ec.visibility_of_all_elements_located((By.CLASS_NAME, "FCS6Q"))
        )
        for opt in options:
            if opt.text == "Remove":
                opt.click()

        self.send_message("Done")
        return True

    def delete_message(self, message: WebElement) -> None:
        if not message:
            return
        try:
            # hover
            el = message.find_element_by_class_name("_3mSPV")
            ActionChains(self.driver).move_to_element(el).perform()
            # click options
            self.driver.find_element_by_xpath(
                '//div[@aria-label="Context Menu"]'
            ).click()
            WebDriverWait(self.driver, 10).until(
                ec.visibility_of_element_located((By.CLASS_NAME, "_1MZM5"))
            )
            # choose delete
            for choice in self.driver.find_elements_by_class_name("_1MZM5"):
                if choice.text == "Delete message":
                    choice.click()
                    sleep(0.1)
                    break
            WebDriverWait(self.driver, 10).until(
                ec.visibility_of_element_located((By.CLASS_NAME, "_1M6AF"))
            )
            # choose delete for everyone
            succeded = False
            for choice in self.driver.find_elements_by_class_name("_1M6AF"):
                if choice.text == "DELETE FOR EVERYONE":
                    choice.click()
                    succeded = True
                    sleep(0.1)
                    break
            if not succeded:
                self.driver.find_element_by_tag_name("html").send_keys(Keys.ESCAPE)
                sleep(0.5)
                self.send_message("This message is too old to be deleted.")
            # scrol back down
            down_btns = self.driver.find_elements_by_xpath(
                '//div[@aria-label="Scroll to bottom"]'
            )
            if down_btns:
                down_btns[0].click()

            self.send_message("Deleted")
        except Exception as e:
            print(f"Erorr: in `delete_message`: {e}")
