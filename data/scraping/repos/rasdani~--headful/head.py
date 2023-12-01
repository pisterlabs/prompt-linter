### instructor has way better undocumented features now
from time import sleep
from openai import OpenAI
import instructor
from enum import Enum
from pydantic import BaseModel, validator
from urllib.parse import urlparse
from typing import Union
from dotenv import load_dotenv

from eyes import see, see_legacy

load_dotenv()
client = instructor.patch(OpenAI())
# GPT_MODEL = "gpt-4-1106-preview"
GPT_MODEL = "gpt-3.5-turbo-1106"


class FunctionName(Enum):
    VISIT_WEBSITE = "visit_website"
    # EXECUTE_BROWSER_ACTION = "execute_browser_action"
    CLICK_WEBPAGE_ELEMENT = "click_webpage_element"


class VisitWebsiteArgs(BaseModel):
    url: str

    @validator("url")
    def check_url(cls, v):
        parsed = urlparse(v)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("Invalid URL")
        return v


# class BrowserAction(Enum):
#     GO_BACK = "go_back"
#     GO_FORWARD = "go_forward"
#     NEW_TAB = "new_tab"


class UserRequest(BaseModel):
    user_request: str


class FunctionCall(BaseModel):
    function_name: FunctionName
    # arguments: Union[VisitWebsiteArgs, ExecuteBrowserActionArgs, UserRequest, None]
    arguments: Union[VisitWebsiteArgs, UserRequest, None]


def visit_website(driver, url):
    driver.navigate_to(url)
    driver.page.wait_for_load_state("networkidle")  # wait for the page to load
    press_f_and_screenshot(driver)
    return driver


# def execute_browser_action(driver, action: BrowserAction):
#     page = driver.page
#     if action == BrowserAction.GO_BACK:
#         page.go_back()
#         page.wait_for_load_state("networkidle")  # wait for the page to load
#         press_f_and_screenshot(driver)
#     elif action == BrowserAction.GO_FORWARD:
#         page.go_forward()
#         page.wait_for_load_state("networkidle")  # wait for the page to load
#         press_f_and_screenshot(driver)
#     elif action == BrowserAction.NEW_TAB:
#         viewport_size = page.viewport_size
#         new_page = page.context.new_page()
#         new_page.set_viewport_size(viewport_size)
#         driver.page = new_page
#     else:
#         raise ValueError(f"Unknown action: {action.action}")
#     return driver


def click_webpage_element(user_request):
    image_path = "screenshot_after.png"
    response = see(image_path=image_path, user_request=user_request)
    # image_paths = [
    #     "screenshot_before.png",
    #     "screenshot_after.png"
    # ]
    # response = see_legacy(image_paths=image_paths, user_request=user_request)
    return response


def press_f_and_screenshot(driver):
    driver.page.keyboard.press("Escape")
    page = driver.page
    # page.screenshot(path="screenshot_before.png", full_page=True)
    driver.take_screenshot(path="screenshot_before.png")
    sleep(2)
    page.keyboard.press("f")
    sleep(2)
    # page.screenshot(path="screenshot_after.png", full_page=True)
    driver.take_screenshot(path="screenshot_after.png")


def execute_function_call(driver, function_call: FunctionCall):
    response = None
    if function_call.function_name == FunctionName.VISIT_WEBSITE:
        url = function_call.arguments.url
        driver = visit_website(driver, url)
    # elif function_call.function_name == FunctionName.EXECUTE_BROWSER_ACTION:
    #     action = function_call.arguments.action
    #     driver = execute_browser_action(driver, action)
    elif function_call.function_name == FunctionName.CLICK_WEBPAGE_ELEMENT:
        user_request = function_call.arguments.user_request
        response = click_webpage_element(user_request)
        confirmation = input(f"Confirm: {response}? (y/n) ")
        if confirmation == "y":
            driver.enter_text(response)
            press_f_and_screenshot(driver)
        else:
            pass
    else:
        raise ValueError(
            f"Error: function {function_call.function_name} does not exist"
        )
    return driver, response


functions = [
    {
        "type": "function",
        "function": {
            "name": "visit_website",
            "description": "Use this function to visit a website requested by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to enter into the browser",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "If the user wants to click on a specific webpage element, pass the user request on as a string to this function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "Click request from the user. To be passed on.",
                    },
                },
                "required": ["user_request"],
            },
        },
    },
]


def function_call_request(messages, functions=functions, model=GPT_MODEL):
    try:
        func_call: FunctionCall = client.chat.completions.create(
            model=model,
            temperature=0,
            response_model=FunctionCall,
            messages=messages,
            functions=functions,
            max_retries=2,
        )
        return func_call
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
