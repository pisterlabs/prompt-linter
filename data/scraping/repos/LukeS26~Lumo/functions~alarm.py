import os
import subprocess
import re
import random
import threading
import signal
import json
import time
import pytz
from datetime import datetime, timezone, timedelta
import calendar
from timezonefinder import TimezoneFinder
import string
import openai

openai.api_key = ""

timezoneFinder = TimezoneFinder()
main_timezone = pytz.timezone("America/New_York")

alarms_file = 'saved_alarms.json' 
alarms_list = []
arm = "./AlarmRC.mp3"

# load the saved alarms in the alarms file into the alarms list
with open(alarms_file, 'r') as fp:
    alarms_list = json.load(fp)

# update the alarms json file upon the addition or removal of an alarm
def update_alarms():
    with open(alarms_file, 'w') as fp:
        json.dump(alarms_list, fp)


def play_sound(file_name):
    args = ["ffplay", "-autoexit", "-nodisp", file_name]
    music_proc = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    out, err = music_proc.communicate()
    is_finished = music_proc.wait()

    print(is_finished)

def check_alarms():
    cur_moment = datetime.now(main_timezone)
    for a in range(len(alarms_list)):
        if (alarms_list[a][0] == cur_moment.hour and alarms_list[a][1] == cur_moment.minute): 

            # check if the alarm is a repeating alarm
            if len(alarms_list[a]) == 3: 
                if (calendar.day_name[cur_moment.weekday()] in alarms_list[a][2]):
                    print("ALARM!")
                    play_sound(arm)
                break 
            
            print(f"ALARM! today is {calendar.day_name[cur_moment.weekday()]}")
            play_sound(arm)
            alarms_list.remove(alarms_list[a]) # remove alarm from list if it's a one-time alarm
            with open(alarms_file, 'w') as fp:
                json.dump(alarms_list, fp)
            break

def alarm_loop():
    while True:
        check_alarms()

        # play the alarm only once 
        cur_second = datetime.now(main_timezone).second
        time.sleep(61 - cur_second) # ensure the alarm starts playing at the very beginning of the minute

threading.Thread(target=alarm_loop).start()