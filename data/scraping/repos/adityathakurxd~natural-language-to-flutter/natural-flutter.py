from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv
import os
import time

from openai import OpenAI

load_dotenv()  # take environment variables from .env.

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Write Flutter code to build a todo application complete with options to mark a task as complete or delete it"
)

run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id="asst_UWGgXNcLsdomCIVFKH1lKWZz",
  instructions="Only write code nothing else."
)

run = client.beta.threads.runs.retrieve(
  thread_id=thread.id,
  run_id=run.id
)

messages = client.beta.threads.messages.list(
  thread_id=thread.id,
)

for message in messages:
    if message.role == "assistant":
        print(message.content)
        for respo in message.content:
            if respo.type == "text":
                text_content = respo.text.value
                if "```" in text_content:
                    flutter_code = text_content.split("```")[1]
                    print(flutter_code)
                



# # Flutter code to be added
# flutter_code = """
# import 'package:flutter/material.dart';

# void main() {
#   runApp(MyApp());
# }

# class MyApp extends StatelessWidget {
#   @override
#   Widget build(BuildContext context) {
#     return MaterialApp(
#       title: 'Flutter Demo',
#       home: Scaffold(
#         appBar: AppBar(
#           title: Text('Hello Flutter'),
#         ),
#         body: Center(
#           child: Text('Welcome to Flutter!'),
#         ),
#       ),
#     );
#   }
# }
# """

# Setup WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Open the webpage
driver.get("https://dartpad.dev/embed-flutter.html")

try:
     # Open DartPad
    driver.get("https://dartpad.dev/embed-flutter.html")

    # Wait for the page to load
    time.sleep(5)

    # Inject the code into the CodeMirror editor using JavaScript
    js_script = """
    var editor = document.querySelector('.CodeMirror').CodeMirror;
    editor.setValue(arguments[0]);
    """
    driver.execute_script(js_script, flutter_code)

finally:
    # Close the browser after a delay
    time.sleep(1000)  # Adjust time as necessary
    driver.quit()
