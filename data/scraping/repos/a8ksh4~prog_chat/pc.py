#!/usr/bin/env python

import openai
import os
import pyaudio
import wave
import argparse
# import keyboard
import pynput.keyboard as keyboard
import numpy as np
import sounddevice as sd

openai.api_key = os.environ['OPENAI_API_KEY']

# Edit
# Run
# Debug
gpt_preamble = ''''''
menu_params = [
    {'name': 'edit function',
     'info': 'modify a specific function rather than the entire file',
     'parameters': {'function name': 'string',}
    },
    {'name': 'copy function',
     'info': 'copy a specific function and all of its code to a duplicate '
             'function. optionally specify where to put the new copied function in the file.',
     'parameters': {'source function name': 'string', 
                    'destination function name': 'string', 
                    'location in file': '[none|below <function name>|above <function_name>|top|bottom]'}
    },
    {'name': 'save as',
      'info': 'save the current state of the file into a new file',
      'parameters': {'new file name': 'string',}
    },
    {'name': 'quit',
     'info': 'exit the editor. saving the file is optional, but the user must be sure if not saving. ',
     'parametirs': {'save file': 'bool', 
                    'are you sure': 'bool'}
    }
]

def parseArgs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', '--file', type=str, 
                        help='Input file', required=True, 
                        action='store')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Print debug messages')
    return parser.parse_args()


def record_audio_with_trigger(trigger_key=keyboard.Key.up):
    recording = False

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if recording:
            audio_data.append(indata)

    with sd.InputStream(callback=callback):
        audio_data = []

        print(f"Press '{trigger_key}' to start recording...")
        is_pressed = False

        def on_key_press(key):
            nonlocal recording, is_pressed
            if key == trigger_key and not is_pressed:
                print("Recording audio...")
                recording = True
                is_pressed = True

        def on_key_release(key):
            nonlocal recording, is_pressed
            if key == trigger_key and is_pressed:
                print("Recording stopped.")
                recording = False
                is_pressed = False

        listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        listener.start()
        listener.join()

    audio_data = np.concatenate(audio_data, axis=0)

    return audio_data

def convert_audio_to_text(audio_data):
    '''Use openai whisper to convert the audio.'''
    pass


if __name__ == '__main__':
    args = parseArgs()
    if args.debug:
        print(args)

    if not os.path.exists(args.file):
        with open(args.file, 'w') as f:
            f.write('#!/usr/bin/env python\n\n')
    
    while True:
        # Show the current file with line numbers
        
        with open(args.file, 'r') as f:
            code = f.read()
        #code = [l.strip() for l in code]
        for n, line in enumerate(code.split('\n')):
            print(f'{n}: {line}')
        print()
        # wait for user to hold space and record audio until they release
        # print('Hold space to record')
        # audio_data = record_audio_with_trigger()
        # converted_text = convert_audio_to_text(audio_data)
        text = input('Enter text: ')
        print()
        instructions = "You will uses the user instructions to modify "\
                       "some python code.  Reply with only python code. "\
                       "Don't drop the shebang line."
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "system", "content": code},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=1024
        )
        print(result)
        new_code = result.choices[0]['message']['content']
        for n, line in enumerate(new_code.split('\n')):
            print(f'{n}: {line}')
        print()
        keep = input('Keep? [y/n]: ').lower() == 'y'
        if keep:
            with open(args.file, 'w') as f:
                f.write(new_code)