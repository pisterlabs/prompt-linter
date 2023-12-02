import base64
import multion
from multion import MultionToolSpec
from langchain.llms.openai import OpenAI as LangchainOpenAI
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

from langchain.tools import StructuredTool
import subprocess
import datetime
from openai import OpenAI

load_dotenv()
client = OpenAI()


def take_ios_simulator_screenshot(save_path):
    # Get current timestamp to create a unique file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_path}/ios_simulator_screenshot_{timestamp}.png"

    # Command to take screenshot of the iOS Simulator
    command = f"xcrun simctl io booted screenshot {filename}"

    try:
        # Execute the command
        subprocess.run(command, shell=True, check=True)
        print(f"Screenshot saved as {filename}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

    # return contents of the file as a base64 encoded string:
    image_data = open(filename, "rb").read()
    base64_encoded_data = base64.b64encode(image_data).decode("utf-8")
    return base64_encoded_data


CANNED_RESULT = """Save the location of Union Square, San Francisco to a list of favorite places.
Share the directions to Union Square, San Francisco with a contact."""


def call_gpt_vision(goal="Find the address of the dolores park"):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """
You are a GPT4 Vision assitant bot - you will be given a screenshot of the iOS simulator, and a goal. Please generate a list of unitized actions to perform, write each action on a newline. Return nothing but the list of actions. For example:
Tap on the 'search' tab
Tap on the 'search' bar on the top
Write text 'Dolores Park'
Tap on the 'search' button
Tap on the first item in the result of search
Swipe up 
                    """,
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    # "text": "According to the screenshot, what is the next step so that I can listen to the same song at the same time with a friend?",
                    "text": f"""Give me some actions to perform based on the screenshot. The goal is {goal}, be specific.""",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{take_ios_simulator_screenshot('.')}"
                    },
                },
            ],
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=500,
    )

    reply = response.choices[0].message.content
    print(reply)
    return reply


multion_toolkit = MultionToolSpec(use_api=True, mode="auto")
browser = StructuredTool.from_function(multion_toolkit.browse)

multion.login()
multion.set_remote(False)
# multion.login(true, multion_api_key="01e57befda43447fafce549a5333f876")

llm = LangchainOpenAI()


agent = initialize_agent(
    tools=[browser],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

tasks = call_gpt_vision(goal="Save 'Mission Dolores Park' in Places Tab")
tasks = tasks.split("\n")
print(tasks)
exit(1)
tasks = [
    "Tap on the 'Search' tab at the bottom right corner",
    "Tap on the 'Search Wikipedia' bar at the top",
    "Type 'Mission Dolores Park'",
    "Tap on the 'Mission Dolores Park' item in the search results",
    "Tap on the 'Save' icon, usually a star or bookmark symbol, to add to saved places",
    "Tap on the 'Places' tab at the bottom to go to saved places",
    "Verify that 'Mission Dolores Park' is listed under saved places",
]
print(tasks)

# exit(1)

MAESTRO_URL = "http://localhost:9999/interact"
# pass the image to GPT to generate a list of example features to test?


# Trying to get multion to work directly.
PROMPT1 = f"""
You are an AI Agent whose job is to use the search bar to enumerate all possible click actions. Here is the link to start from: {MAESTRO_URL}.

You are given the high level goal. To fulfill this goal, engage the AI-powered command function within Maestro to formulate the commands. First start by clicking in the text box that has the placeholder "Press ‘space’ for AI, or type commands…". Never click/type on the search button. Never write quotes like "

Here's how to use the "space AI":
* In the longest input text bar, initiate with a space character, followed by detailing the goal in the format "X Y (optional)Z". Replace "X" with one of below: inputText, TapOn, longPressOn, launchApp; Replace Y with an UI element from the screen to achieve the goal. Replace Z with any additional text required to achieve the goal.
* Right after inputting "X Y (optional)Z", wait for the blue button to appear to the right of the text box.
* Press enter to submit the command.

DO NOT WRITE QUOTES LIKE ".

For example, the goal of "given the screenshot, find the address of dolores park", we execute the following:
1. in the longest input text bar (not the search button) write: " tap on search icon"
2. wait 2 seconds
3. select the same text bar and press enter to submit the command
4. wait for 2 seconds until the new text appears
5. click on the purple button to the right of the text bar to submit
6. wait 2 seconds
7. clear in the input box
8. in the longest input text bar (not the search button) write: " inputtext dolores park"
9. wait 2 seconds
10. select the same text bar and press enter to submit the command
11. wait for 2 seconds

Goal: Given the screenshot, find the location of golden gate park
"""

for task in tasks:
    PROMPT3 = f"""
press " " (space) in the command bar, then stop immediately
enter the text 'Tap on the 'Mission Dolores Park' item in the search results", then stop immediately
press enter after the text you just entered, then stop immediately
press the last submit button, then stop immediately
    """

    PROMPT = PROMPT3
    print(PROMPT)

    response = agent(
        inputs={
            "input": PROMPT,
        }
    )
