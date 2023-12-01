import asyncio
import time
import sqlite3
import hashlib
from pygame import mixer
import elevenlabs as eleven
import pygame._sdl2 as sdl2
from gtts import gTTS
import os
import re
import random
import signal
import openai
import speech_recognition as sr
import params

from viam.components.servo import Servo
from viam.components.base import Base
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient
from viam.rpc.dial import Credentials, DialOptions

# set up db cache
con = sqlite3.connect("cache/rosey_cache.db")
cur = con.cursor()
res = cur.execute("SELECT name FROM sqlite_master WHERE name='responses'")
if res.fetchone() is None:
    cur.execute("CREATE TABLE responses(prompt, char, mood, response)")

openai.organization = params.openai_organization
openai.api_key = params.openai_api_key
if (params.elevenlabs_key):
    eleven.set_api_key(params.elevenlabs_key)

# if you want to specify a specific device, you can pass devicename = params.mixer_device
# see https://pypi.org/project/SpeechRecognition/ for troubleshooting tips
mixer.init()
robot = ''
current_char = ""
current_mood = ""
current_person_name = ""

async def connect():
    opts = RobotClient.Options.with_api_key(
      api_key=params.viam_api_key,
      api_key_id=params.viam_api_key_id
    )
    return await RobotClient.at_address(params.viam_address, opts)

async def say(text):
    file = 'cache/' + current_char + hashlib.md5(text.encode()).hexdigest() 
    try:
        if (re.match("^As an AI language model", text) or not os.path.isfile(file + ".mp3")):
            if (params.elevenlabs_key and ((current_char and "voice" in params.char_list[current_char.lstrip()]) or (params.elevenlabs_default_voice != ""))):
                if (current_char and "voice" in params.char_list[current_char.lstrip()]):
                    print(params.char_list[current_char.lstrip()])
                    voice = params.char_list[current_char.lstrip()]["voice"]
                else:
                    voice = params.elevenlabs_default_voice
                audio = eleven.generate(text=text, voice=current_char)
                time.sleep(1)
                eleven.save(audio=audio, filename=file + ".mp3")
                time.sleep(1)
            else:
                myobj = gTTS(text=text, lang='en', slow=False)
                myobj.save(file + ".mp3")
        mixer.music.load(file + ".mp3") 
        mixer.music.play() # Play it

        while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)
    except RuntimeError:
        await say("nevermind")

async def make_something_up(seen):
    tones = ['angry', 'happy', 'sad']
    prefix = {'angry': ['Really? Is that a', 'No! I see a', 'Shoot, its a'], 
        'happy': ['Well look, its a', "Yes! I see a", "Yay, its a"], 'sad': ['Oh no, its a', 'I fear I see a', 'Sadly, I think thats a']}
    
    if current_mood == "":
        chosen_tone = random.choice(tones)
    else:
        chosen_tone = current_mood

    command = "say a short " + chosen_tone + " " + random.choice(params.completion_types) + " about a " + ' and a '.join(seen)
    seen_sentence = "say '" + current_person_name + "," + random.choice(prefix[chosen_tone]) + " " + ' and a '.join(seen) + "'"
    print(seen_sentence)
    print(command)
    await move_servo(chosen_tone)    
    await say(await ai_command(seen_sentence))
    resp = await ai_command(command)
    resp = re.sub('Q:',  'Question: ', resp)
    resp = re.sub('A:',  'Answer: ', resp)
    await say(resp)

async def ai_command(command):
    # get cached response if it exists
    res = cur.execute("SELECT response FROM responses WHERE prompt=? and char=? and mood=?", (command, current_char, current_mood))
    response = res.fetchone()
    if(response):
        return response[0]
    try:
        if (current_char != ""):
            style = current_char
        else:
            # default to "a companion" to avoid "As an AI language model..."
            style = "a companion"
        
        command_update = "give me a quote that a " + current_mood + " " + style + " might say in response to " + command
        print(command_update)
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", max_tokens=1024, messages=[{"role": "user", "content": command_update}])
        completion = completion.choices[0].message.content
        print(completion)
        completion = re.sub('[^0-9a-zA-Z.!? ]+', '', completion)
        print(completion)
        if(not re.match("^As an AI language model", completion)):
            # cache this completion
            cur.execute("INSERT into responses VALUES(?, ?, ?, ?)", (command, current_char, current_mood, completion))
            con.commit()
        return completion
    except openai.error.ServiceUnavailableError:
        errors = ["Sorry, I am feeling tired", "Sorry, I forget", "Never mind, I don't know"]
        return random.choice(errors)

async def move_servo(pos):
    if params.enable_emotion_wheel == True:
        pos_angle = {
                    "happy": 0,
                    "angry": 75,
                    "sad": 157
                }
        service = Servo.from_robot(robot, 'servo1')
        await service.move(angle=pos_angle[pos])

async def see_something():
    service = VisionClient.from_robot(robot, 'vis-stuff-detector')
    found = False
    count = 0
    while not found:
        detections = await service.get_detections_from_camera(camera_name='cam')
        # if you are using a classifier model instead of detector...
        # detections = await service.get_classifications_from_camera(camera_name='cam', count=1)
        for d in detections:
            if d.confidence > params.vision_confidence:
                print(detections)
                if d.class_name != '???':
                    found = True
                    await make_something_up([d.class_name])
            count = count + 1
            if count > 20:
                found = True
                # nothing significant seen, so stop trying
                await say("nothing")

async def mood_motion(base, mood):
    if mood == "sad":
        await base.move_straight(distance=-100, velocity=150)
    elif mood == "happy":
        await base.spin(angle=-10, velocity=1000)
        await base.spin(angle=10, velocity=1000)
        await base.spin(angle=-10, velocity=1000)
        await base.spin(angle=10, velocity=1000)
    elif mood == "angry":
        await base.move_straight(distance=50, velocity=1000)
        time.sleep(.05)
        await base.move_straight(distance=-20, velocity=1000)
        time.sleep(.05)
        await base.move_straight(distance=50, velocity=1000)

async def loop(robot):
    base = Base.from_robot(robot, 'viam_base')
    r = sr.Recognizer()
    r.energy_threshold = 1568 
    r.dynamic_energy_threshold = True
    m = sr.Microphone()
    await move_servo("happy")

    print("Setup complete, listening...")
    while True:
        with m as source:
            r.adjust_for_ambient_noise(source) 
            audio = r.listen(source)
        try:
            global current_char
            global current_mood
            global current_person_name
            transcript = r.recognize_google(audio_data=audio, show_all=True)
            if type(transcript) is dict and transcript.get("alternative"):
                text = transcript["alternative"][0]["transcript"].lower()
                print(text)
                if re.search(".*" + params.robot_command_prefix, text):
                    command = re.sub(".*" + params.robot_command_prefix + "\s+",  '', text)
                    print(command)
                    if command == "spin":
                        await base.spin(angle=720, velocity=500)
                    elif command == "turn a little right":
                        await base.spin(angle=-45, velocity=500)
                    elif command == "turn right":
                        await base.spin(angle=-90, velocity=500)
                    elif command == "turn a little left":
                        await base.spin(angle=45, velocity=500)
                    elif command == "turn left":
                        await base.spin(angle=90, velocity=500)
                    elif command == "turn around":
                        await base.spin(angle=180, velocity=500)
                    elif command == "move forward":
                        await base.move_straight(distance=1000, velocity=500)
                    elif command == "move backwards":
                        await base.move_straight(distance=-1000, velocity=500)
                    elif command == "reset":
                        current_char = ""
                        current_mood = ""
                        current_person_name = ""
                    elif re.search("^" + '|'.join(params.observe_list), command):
                        await see_something()
                    elif command == "act random":
                        current_char = random.choice(params.char_list.keys())
                        await say(await ai_command("Say hi " + current_person_name))
                    elif re.search("^" + params.intro_command, command):
                        current_person_name = re.sub(params.intro_command, "", command)
                        await say(await ai_command("Say hi " + current_person_name))
                    elif re.search("^" + params.char_command +" (" + '|'.join(params.char_list.keys()) + ")", command):
                        current_char = re.sub(params.char_command, "", command)
                        await say(await ai_command("Say hi"))
                    elif re.search("^" + params.char_guess_command +" (" + '|'.join(params.char_list.keys()) + ")", command):
                        if current_char != "":
                            char_guess = re.sub(params.char_guess_command, "", command)
                            print("guess: |" + char_guess + "|actual: |" + current_char + "|")
                            if char_guess == current_char:
                                await say(await ai_command("say 'You are correct'"))
                            else:
                                await say(await ai_command("say 'You are wrong, try again'"))
                    elif re.search("^you (seem|look)", command):
                        current_mood = re.sub("you (seem|look) ", "", command)
                        asyncio.gather(
                            await mood_motion(base, current_mood),
                            await move_servo(current_mood),
                            await say(await ai_command("Say 'yeah, I am " + current_mood +  "'")),
                        )
                    else:
                        await say(await ai_command(command))

        except sr.UnknownValueError:
            print("Speech recognition could not understand audio, trying again")
        except Exception as e:
            print("Exception while running loop")
            raise e

async def main():
    global robot
    robot = await connect()
    print("Connected to robot...")
    try:
        await loop(robot=robot) 
    finally:
        print("Stopping...")
        try:
            await robot.close()
        except asyncio.CancelledError:
            # can be safely ignored
            pass

if __name__ == '__main__':
    asyncio.run(main())
