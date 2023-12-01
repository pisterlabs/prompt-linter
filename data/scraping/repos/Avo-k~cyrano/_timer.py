import time
import threading
import pyttsx3
from langchain.tools import tool
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class Timer:
    def __init__(self, duration):
        self.duration = duration
        self.start_time = None
        self.remaining = duration
        self.thread = None
        self.paused = False
        self.stopped = False

    def start(self):
        self.start_time = time.time()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        while not self.stopped and self.remaining > 0:
            if not self.paused:
                time.sleep(1)
                self.remaining = self.duration - (time.time() - self.start_time)
        if self.stopped:
            print("Timer stopped")
        else:
            print("Timer finished")
            engine = pyttsx3.init()
            engine.say("dring ! dring! les pâtes sont prêtes !")
            engine.runAndWait()  # play the alarm

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        self.start_time = time.time() - (self.duration - self.remaining)

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join()


# @tool
# def start_timer(seconds):
#     """Start a timer for a given duration

#     Args:
#         seconds (int): number of seconds to run the timer for.
#     """
#     duration = int("".join(filter(str.isnumeric, seconds)))
#     assert isinstance(duration, int)
#     timer = Timer(duration=duration)
#     timer.start()
#     return "minuteur lancé"


class SecondsInput(BaseModel):
    """Inputs for start_timer"""

    seconds: int = Field(description="Duration of the timer in seconds")


class TimerTool(BaseTool):
    name = "start_timer"
    description = "Useful when you want to start a timer."
    args_schema: Type[BaseModel] = SecondsInput

    def _run(self, seconds: int) -> str:
        timer = Timer(duration=seconds)
        timer.start()
        return "minuteur lancé"

    def _arun(self, seconds: int):
        raise NotImplementedError("start_timer does not support async")
