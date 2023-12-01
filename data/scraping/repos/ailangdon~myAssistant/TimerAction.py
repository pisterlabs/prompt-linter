import re
from langchain.schema import SystemMessage, HumanMessage

from Action import Action
import time
import threading
import beepy

def timer_callback(timeout):
    time.sleep(timeout)
    for i in range(3):
        time.sleep(2)
        beepy.beep(sound=5)
    beepy.beep(sound=7)

class TimerAction(Action):
    def __init__(self, llm):
        self.name="timer"
        self.selection_prompt="Set timer to a duration?"
        self.parameter_prompt="Task: What is the duration in seconds that should be waited? Just answer with the number of seconds, Don't give an explanation, don't specify the unit. Remember 1 minute is 60 seconds. Seconds:"
        self.llm = llm

    def execute(self, query):
        timeout = self.get_single_paramter(query)
        if timeout > 0:
            self.run_timer(timeout)
            return "Timer set to "+str(timeout)+" seconds."
        else:
            return "Sorry, I didn't understand for how long I should set the timer."

    def get_single_paramter(self, query):
        prompt = query+" "+self.parameter_prompt
        messages = [
            SystemMessage(content = "You are a helpful assistant who sets a timer to remember the user after a specific time."),
            HumanMessage(content = prompt)
        ]
        response = self.llm(messages)
        print("Time extraction response: "+str(response)+ " "+str(response.content))
        timeout_in_seconds = self.parse_number(response.content)
        return timeout_in_seconds

    def parse_number(self, text):
        # Regular expression to match the first number in the string (with commas as thousand separators)
        pattern = r"\b\d{1,3}(?:,\d{3})*\b"

        # Search for the first number in the string using the regular expression
        match = re.search(pattern, text)

        # Extract the matched number if found
        if match:
            extracted_number = float(match.group().replace(',', ''))
            return extracted_number
        else:
            return -1

    def run_timer(self, timeout):
        timer_thread = threading.Thread(target=timer_callback, args=(timeout,))
        timer_thread.start()