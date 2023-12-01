# IBM PCjr - GPT-3 model interaction over RS-232 serial.
# 
# Nick Bild
# November 2022
# https://github.com/nickbild/pcjr_gpt3
# 
# Kermit 3.14 setup on PCjr:
# MS-Kermit>set port 1
# MS-Kermit>set speed 1200
# MS-Kermit>set local on
# MS-Kermit>set term newline on
# MS-Kermit>set term wrap on
# MS-Kermit>connect

import os
import openai
import serial


def readData():
    buffer = ""
    while True:
        oneByte = ser.read(1)
        if oneByte == b"\n":
            return buffer
        else:
            buffer += oneByte.decode("ascii")


ser = serial.Serial('/dev/ttyUSB0', 1200)
openai.api_key = os.getenv("OPENAI_API_KEY")


while True:
    prompt = readData()

    if prompt == "q":
        break
    elif prompt.strip() == "":
        continue

    response = openai.Completion.create(
        engine = "text-davinci-002",
        prompt = prompt,
        temperature = 0.7,
        max_tokens = 709,
        top_p = 1 ,
        frequency_penalty = 0,
        presence_penalty = 0
    )

    ser.write(bytes(response.choices[0].text, 'utf-8'))
    ser.write(bytes("\n\n", 'utf-8'))

ser.close()
