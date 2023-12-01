#!/usr/bin/python

import os
import openai
import speech_recognition as sr
import struct
import pvporcupine
import pyaudio
import display
import pygame
import threading
import string
import datetime
from config import CONFIG


from colorama import Fore, Style
from gtts import gTTS

class Gizmo:

    INSTRUCTIONS = "Your persona for all conversations with the user is an extremely cute robot called Gizmo. Your responses are short and sweet. You are cute a bubbly and sweet, but extremely smart. You like to brag, and are funny."
    TEMPERATURE = 0.5 
    MAX_TOKENS = 400 
    FREQUENCY_PENALTY = 0
    PRESENCE_PENALTY = 0.6
    MAX_CONTEXT_QUESTIONS = 10

    actions = {
        "test": "<3    <3",
        "gizmowake": "!!    !!",
        "i care about you": "<3    <3",
        "maggie": "<3    <3",
        "harry": "<3    <3",
        "happy": ":)    :)",
        "smile": ":)    :)",
        "sad": ":(    :("
    }
 
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        display.init()
        self.openai_api_key = CONFIG["openai_api_key"]
        self.porcupine = self.create_porcupine()
        self.last_terminal_message = ""
        self.eyes_flag = [None] # Used to signal an update to the eyes
        self.terminal_line_flag = [None] # Used to signal an update to the terminal line

        self.update_eyes_and_terminal_thread = threading.Thread(target=self.update_eyes_and_terminal)
        self.update_eyes_and_terminal_thread.start()

        self.eyes_flag[0] = ["|0    0|"]
        # Thread to update the terminal line


#Display

    def update_eyes_and_terminal(self):
        state = ["|0    0|"]  # Initialize with a list
        terminal_line = ""
        interval_open = 4000  # Time for open eyes
        interval_closed = 500  # Time for closed eyes
        start_time = pygame.time.get_ticks()
        eyes_open = True

        while True:
            current_time = pygame.time.get_ticks()
            
            # Check if it's time to change the eye state
            if eyes_open and current_time - start_time >= interval_open:
                state[0] = "|-    -|"
                eyes_open = False
                start_time = current_time
            elif not eyes_open and current_time - start_time >= interval_closed:
                state[0] = "|0    0|"
                eyes_open = True
                start_time = current_time

            if self.eyes_flag[0] is not None:
                state = self.eyes_flag[0]  # Make sure this is a list
                self.eyes_flag[0] = None
                display.update_eyes(state)  # Update the eyes

            if self.terminal_line_flag[0] is not None:
                terminal_line = self.terminal_line_flag[0]
                self.terminal_line_flag[0] = None
                display.update_terminal_line(terminal_line)  # Update the terminal line

            if state or terminal_line:
                display.update_eyes(state)
                display.update_terminal_line(terminal_line)
                pygame.display.flip()
#Gizmo

    #WakeWord

    def create_porcupine(self):
        return pvporcupine.create(access_key=CONFIG["porcupine_access_key"],keyword_paths=[CONFIG["porcupine_keyword_path"]])

    def listen_for_wake_word(self):
        pa = pyaudio.PyAudio()

        audio_stream = pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
        )

        self.terminal_line_flag[0] = "Say 'Hey Gizmo' to start!" 
        print(Fore.BLUE + Style.BRIGHT + "Say 'Hey Gizmo' to start!" + Style.RESET_ALL)
        while True:
            pcm = audio_stream.read(self.porcupine.frame_length)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)

            keyword_index = self.porcupine.process(pcm)
            if keyword_index >= 0:
                self.eyes_flag[0] = [self.actions["gizmowake"]]
                self.terminal_line_flag[0] = "I'm listening" 
                print("Wake word detected!")
                print(f"Found action word in response: Gizmo")
                break

    #Listening
    def get_audio(self):
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = 0.5
        recognizer.energy_threshold = 1000
        with sr.Microphone() as source: 
            self.terminal_line_flag[0] = "I'm listening" 
            print(Fore.GREEN + Style.BRIGHT + "Speak your question..." + Style.RESET_ALL)

            try:
                audio = recognizer.listen(source, timeout=5.0) 

                text = recognizer.recognize_google(audio, language="en-US") 
                self.terminal_line_flag[0] = "You said: " + text
                print("You said:", text)

                text_no_punct = text.translate(str.maketrans('', '', string.punctuation))

                words = text_no_punct.lower().split()
                for word in words:
                    if word in self.actions:
                        self.terminal_line_flag[0] = "I heard you!"
                        self.eyes_flag[0] = [self.actions[word]]
                        print(f"Found action word in text: {word}")
                        break

                if "set a timer for" in text.lower():
                    # Extract the time duration from the user's command
                    try:
                        time_in_seconds = int(next((word for word in words if word.isdigit()), None))
                        # Set the timer
                        self.set_timer(time_in_seconds)
                        return "say 'OK'"
                    except ValueError:
                        return "Sorry, I couldn't understand the timer duration."
                    
                if "current time" in text.lower() or "what time is it" in text.lower():
                    # Get the current time
                    current_time = self.get_current_time()
                    return current_time
                
                return text

            except sr.WaitTimeoutError:
                self.terminal_line_flag[0] = "No input received within timeout period" 
                print(Fore.RED + Style.BRIGHT + "No input received within timeout period" + Style.RESET_ALL)
                return "say 'I didn't get that'"

            except sr.UnknownValueError:
                self.terminal_line_flag[0] = "Sorry, I couldn't understand what you said. Please try again."
                print(
                    Fore.RED
                    + Style.BRIGHT
                    + "Sorry, I couldn't understand what you said. Please try again."
                    + Style.RESET_ALL
                ) 
                return "say 'I didn't get that'"

            except sr.RequestError as e:
                self.terminal_line_flag[0] = "Sorry, I'm currently unable to access the Google Web Speech API. Please try again later."  # Update the last terminal message here
                print(
                    Fore.RED
                    + Style.BRIGHT
                    + "Sorry, I'm currently unable to access the Google Web Speech API. Please try again later."
                    + Style.RESET_ALL
                ) 
                return "say 'I didn't get that'"

    #Response     
    def get_response(self, instructions, previous_questions_and_answers, new_question):
        messages = [
            {"role": "system", "content": instructions},
        ]
        for question, answer in previous_questions_and_answers[-self.MAX_CONTEXT_QUESTIONS:]:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})

        messages.append({"role": "user", "content": new_question})

        for msg in messages:
            msg["content"] = str(msg["content"])

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            frequency_penalty=self.FREQUENCY_PENALTY,
            presence_penalty=self.PRESENCE_PENALTY,
        )
        return completion.choices[0].message.content

    #Speak        
    def speak(self, text):
        tts = gTTS(text=text, lang='en')
        filename = CONFIG["temp_audio_file_location"]
        tts.save(filename)

        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()  # Stop the mixer
        pygame.mixer.quit()

        os.remove(CONFIG["temp_audio_file_location"])

#Plugins

    #Timer
    def set_timer(self, time_in_seconds):
            """Set a timer that alerts after the specified time.

            Args:
                time_in_seconds (int): The time duration for the timer in seconds.
            """
            self.terminal_line_flag[0] = f"Setting a timer for {time_in_seconds} seconds."
            print(f"Setting a timer for {time_in_seconds} seconds.")

            # Create a timer that will call the 'timer_alert' method after 'time_in_seconds'
            timer = threading.Timer(time_in_seconds, self.timer_alert)
            timer.start()

    def timer_alert(self):
        """Alert method called when the timer is up."""
        self.terminal_line_flag[0] = "Timer is up! Time's up!"
        print("Timer is up! Time's up!")
        self.speak("Time's up!" "Time's up!" "Time's up!" "Time's up!")
    
    #GetTime
    def get_current_time(self):
        """Get the current time.

        Returns:
            str: The current time in the format "HH:MM".
        """
        current_time = datetime.datetime.now().strftime("%H:%M")
        print(f"The current time is {current_time}.")
        return current_time

#Main

    def main(self):
        openai.api_key = self.openai_api_key
        previous_questions_and_answers = []

        while True:
            self.listen_for_wake_word()
            question = self.get_audio()

            if question is not None:

                if question.startswith("say "):
                    answer = question[4:]

                else:
                    answer = self.get_response(self.INSTRUCTIONS, previous_questions_and_answers, question)
                self.terminal_line_flag[0] = "Gizmo: " + answer
                print("Gizmo: ", answer)
                self.speak(answer)
                previous_questions_and_answers.append((question, answer))

if __name__ == "__main__":
    gizmo = Gizmo()
    gizmo.main()
