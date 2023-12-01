import openai
import sys
import os
import pyttsx3
import requests
from datetime import datetime
from weatherconfig import get_weather_info
from facial_expression import detect_expression
from bs4 import BeautifulSoup
import random
from mail import get_num_unread_emails, get_subject_lines_unread_emails
from genderize import Genderize
import webbrowser
from web import play_youtube_video
from image_creation import generate_image
import speech_recognition as sr
import threading
import pyautogui
from stockplotter import plot_stock_data
import torch
import sounddevice as sd
import pyaudio
from num2words import num2words
import subprocess
import json
import numpy as np
from pathlib import Path
from object_detection import object_detection
from dotenv import load_dotenv
load_dotenv()



class Jarvis:
    """
    A personal assistant modeled after Jarvis from Iron Man.

    Attributes:
    - openai: The OpenAI library for creating chat responses.
    - name: The name of the user.
    - previous_interactions: A dictionary of the user's previous interactions.
    - messages: A list of the user's messages.
    - personality: The personality of Jarvis.
    - salutation: The salutation to address the user.
    - engine: The text-to-speech engine.
    - greeting_messages: A dictionary of greeting messages.
    - weather_api_endpoint: The API endpoint for the weather.
    - weather_api_key: The API key for the weather.
    - city_name: The name of the city for the weather.
    - latitude: The latitude of the user's location.
    - longitude: The longitude of the user's location.
    - openai_api_key: The API key for OpenAI.
    - genderize: The Genderize library for determining the user's gender.
    """
    
    def __init__(self):
        self.openai = openai
        self.name = os.getenv('YOUR_NAME')
        self.previous_interactions = {}
        self.messages = []
        self.jarvis_personality = f"You are a personal assistant named Jarvis. Your task is to respond to {self.name}'s requests as if you were his personal assistant. Your personality is modeled after the helpful and efficient Jarvis from the Iron Man movies, but you are also capable of adapting your tone to match your user's preferences and needs and should sound natural and conversational. If your response requires the use of numbers, please write out the word instead of using the actual numerical value."
        self.genderize = Genderize()
        self.gen_info = self.genderize.get([self.name])[0]
        if self.gen_info['gender'] == 'male':
            self.salutation = 'Sir'
        else:
            self.salutation = 'ma\'am'
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 190)
        self.greeting_messages = {
            "message1": f"Yes, {self.name}?",
            "message2": f"How can I be of service, {self.salutation}?",
            "message3": f"At your service, {self.salutation}.",
            "message5": f"At your command, {self.name}.",
            "message6": f"Ready and waiting, {self.salutation}.",
            "message7": f"How can I help?"
            }
        
        self.weather_api_endpoint = "https://api.openweathermap.org/data/2.5/forecast"
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.city_name = os.getenv('CITY_NAME')
        self.lat = os.getenv('YOUR_LATITUDE')
        self.lon = os.getenv('YOUR_LONGITUDE')
        self.openai.api_key = os.getenv('OPEN_AI_APIKEY')
        self.email_password = os.getenv('GMAIL_PASS')
        self.email = os.getenv('GMAIL_EMAIL')
        self.supbox = []
        
        

    def process_input(self, input_text):
        """Process user input and return an AI-generated response.

        Args:
            input_text (str): The user input to be processed.

        Returns:
            str: The AI-generated response to the user input.

        This method adds the user input to a list of previous messages and uses OpenAI's
        GPT-3.5 Turbo model to generate a response based on the entire conversation history.
        The response is then added to the message list and returned.
        """

        try:
            if self.messages[-1]['role'] != "assistant":
                
                self.messages.append({"role": "assistant", "content": f"{self.jarvis_personality}"})
        except IndexError:
            self.messages.append({"role": "assistant", "content": f"{self.jarvis_personality}"})
        
        self.messages.append({f"role": "user", "content": f"{input_text}"})
        
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=self.messages
        )
        
        if len(self.supbox) > 0:
            reply = "Do you need anything else?"
            self.supbox.clear()
               
        else:
            reply = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": reply})
            
        return reply
    
    
    def get_random_greeting(self):
        """
        Select a random greeting message from the dictionary of greeting messages.

        The method chooses a random key from the `greeting_messages` attribute,
        which is assumed to be a dictionary mapping strings to strings. It then
        returns the corresponding value associated with the selected key.

        Returns:
            A string representing a randomly selected greeting message.
        """
        self.random_key = random.choice(list(self.greeting_messages.keys()))
        self.random_value = self.greeting_messages[self.random_key]
        return self.random_value
    
    def say(self, text):
        """
        Uses the text-to-speech engine to speak the provided text.

        Args:
            text (str): The text to be spoken.

        Returns:
            None
        """
        # device = torch.device('cuda')
        # torch.set_num_threads(8)
        # local_file = 'models\\v3_en.pt'

        # model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
        # model.to(device)

        # example_text = f'{text}'
        # sample_rate = 48000
        # speaker='en_103'

        # # Define the PyAudio stream and callback function
        # pa = pyaudio.PyAudio()
        # stream = pa.open(format=pyaudio.paFloat32,
        #                 channels=1,
        #                 rate=sample_rate,
        #                 output=True)

        # # Define the cache file path
        # cache_path = Path('audio_cache.json')

        # # Define the cache dictionary
        # audio_cache = {}

        # # Load the cache from the file or from memory
        # if cache_path.exists():
        #     with open(cache_path, 'r') as f:
        #         for line in f:
        #             # Parse each line of the JSON file and add it to the cache
        #             data = json.loads(line)
        #             audio_cache.update(data)
        # else:
        #     audio_cache = {}

        # # Define the audio generation function
        # def generate_audio(text):
        #     if text in audio_cache:
        #         # Use cached audio if available
        #         audio = np.array(audio_cache[text], dtype=np.float32)
                
        #     else:
        #         # Generate audio using the PyTorch model
        #         with torch.no_grad():
        #             audio = model.apply_tts(text=text,
        #                                     speaker=speaker,
        #                                     sample_rate=sample_rate,
        #                                     put_accent=True,
        #                                     put_yo=True).cpu().numpy()
        #         # Add generated audio to the cache
        #         audio_cache[text] = audio.tolist()
        #         with open(cache_path, 'a') as f:
        #             # Write the new data to the end of the JSON file
        #             json.dump({text: audio.tolist()}, f)
        #             f.write('\n')  # Add a newline character to separate entries

        #     return audio

        # # Generate and stream the audio
        # with torch.no_grad():
        #     audio = generate_audio(example_text)
        # stream.start_stream()
        # stream.write(audio.tobytes())
        # stream.stop_stream()
        # stream.close()
        # pa.terminate()
        self.engine.say(text)
        self.engine.runAndWait()

 
        
    def get_user_input(self):
        """
        Captures audio input from the user's microphone and returns the recognized text.

        Returns:
        --------
        str or None:
            Returns the recognized text if successful, otherwise None.

        Raises:
        -------
        None.

        Example:
        --------
        To capture user input and get the recognized text, call the method as follows:
            recognized_text = get_user_input()
        """

        self.r = sr.Recognizer()
        
        with sr.Microphone() as source:

            try:
                self.user_choice_audio = self.r.listen(source, timeout=10)
            except sr.WaitTimeoutError:
                self.say(f"Goodbye, just call my name if you need anything else")
                return None
            except sr.RequestError:
                self.say(f"Sorry {self.salutation}, I'm having trouble accessing the microphone.")
                return None

        try:
            self.user_choice = self.r.recognize_google(self.user_choice_audio).lower()
            return self.user_choice
        except sr.UnknownValueError:
            self.say("Goodbye, just call my name if you need anything else")
            return None
        except sr.RequestError:
            self.say("Sorry, I'm having trouble accessing the internet right now.")
            return None
                
                
    def get_weather(self):
        """
        Retrieves current weather data from OpenWeatherMap API based on the latitude and longitude provided by the user.

        If "weather" is included in the user's choice, the function constructs a URL for the OpenWeatherMap API and sends a GET request
        to retrieve the weather data. The data is stored in self.weather_data and a message containing the data is appended to self.messages.

        Args:
            None

        Returns:
            None
        """

        if "weather" in self.user_choice:
            self.weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={self.lat}&lon={self.lon}&appid={self.weather_api_key}"
            response = requests.get(self.weather_url)
            self.weather_data = response.json()
            self.messages.append({"role": "assistant", "content": f"Here is the all the relevant current weather data. i am going to ask you questions about it and you respond accordingly. If your response requires the use of numbers, please write out the word instead of using the actual numerical value.. Current Weather Data: {get_weather_info(self.weather_data)}."})
            
    def check_mood(self):
        """
        Check the user's mood and prompt a conversation about it.

        This method first checks whether the user has mentioned their mood or feeling
        in their input, by looking for the words "mood" or "feeling" in `self.user_choice`.
        If so, it calls the `detect_expression` function to detect the user's facial
        expression and stores the result in `self.emotion`. Then it adds a message to the
        `self.messages` list, with the assistant's role and a prompt to start a conversation
        about the user's mood.

        Returns:
            None
        """
        if "mood" in self.user_choice or "feeling" in self.user_choice:
            self.emotion = detect_expression()
            self.messages.append({"role": "assistant", "content": f"{self.jarvis_personality} now that you know who you are. here is {self.name}'s current mood is {self.emotion}. i am going to ask you questions about it and you respond accordingly. If your response requires the use of numbers, please write out the word instead of using the actual numerical value."})
        

    def check_stock_market(self):
        """
        Check if the user has requested information about the stock market today.
        If so, fetch the latest financial headlines from Yahoo Finance and add a message to the
        conversation with the headlines, asking the assistant to summarize the information in a short response.

        Returns:
            None
        """
        #check if user said stock market today
        if "stock market" in self.user_choice:
            url = "https://finance.yahoo.com/"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            headlines = [headline.text.strip() for headline in soup.find_all("h3")]
            self.messages.append({"role": "assistant", "content": f"{self.jarvis_personality} now that you know who you are. using these financial headlines aggregate them and answer {self.name}'s request accordingly and your answers should be somewhat short and don't speak of the headlines. If your response requires the use of numbers, please write out the word instead of using the actual numerical value.. Headlines: {headlines}"})  
        
    def check_emails(self):
        """
        Check the user's email account for unread messages, and append a message to the `messages` list
        with the number of unread emails.

        Args:
            self (Jarvis): The current Jarvis instance.

        Returns:
            None
        """
        if "do"in self.user_choice and "emails" in self.user_choice:
            unread = get_num_unread_emails(email=self.email, password=self.email_password)
            self.say(f"You currently have {unread} unread emails")
            self.supbox.append("a")
            
    def read_subject_email(self):
        """
        Reads the subject lines of unread emails in the user's inbox and adds them to the `messages` list.

        If the user's choice includes the phrase "read emails", this function calls the `get_subject_lines_unread_emails()`
        function to retrieve the subject lines of all unread emails in the user's inbox. It then creates a message object
        and adds it to the `messages` list with the `role` field set to "assistant" and the `content` field set to a string
        containing the Jarvis personality and the subject lines of the unread emails.

        Returns:
            None.
        """
        if "read"in self.user_choice and "emails" in self.user_choice: 
            read = get_subject_lines_unread_emails()
            self.messages.append({"role": "assistant", "content": f"{self.jarvis_personality} now that you know who you are. answer the request using this info: in {self.name}'s inbox here are the subject lines of the emails {read}"})

    
    def web_search(self):
        """Searches for a website specified by the user and opens it in a new browser tab.

        If the user's input contains the phrases "pull," "up," and "website," the function prompts the user to provide the URL of the website they want to open. It then uses the OpenAI GPT-3 API to generate a response to the user and extract the URL from the response. Finally, it opens the website in a new browser tab using the `webbrowser` module.

        Returns:
            None
        """
        if "pull" in self.user_choice and "up" in self.user_choice and "website" in self.user_choice:
            define_gpt_web = "Your task is to give me the whole url of the website i am talking about. Don't explain anything i just want the url that is it. nothing before or after it"
            self.say(f"Pulling it up now")
    
            web_list = []
            web_list.append({"role": "assistant", "content": f"{define_gpt_web}"})
            web_list.append({f"role": "user", "content": f"{self.user_choice}"})
        
            title_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=web_list
            )
            web_url = title_response.choices[0].message.content
            self.supbox.append("a")
            def open_website(url):
             webbrowser.open_new_tab(url)
            open_website(web_url)

        
    def morning_protocol(self):
        """
        Initiates the morning protocol by opening several URLs in the user's web browser.

        If the user's choice includes the strings "initiate," "morning," and "protocol," this method
        will initiate the morning protocol by opening several URLs in the user's web browser.

        The URLs that are opened are stored in the `morning_protocol_urls` list. This list can be
        modified to include additional or different URLs as needed.

        This method calls the `morning_web_protocol` function, which accepts a list of URLs as an argument
        and opens each URL in a new tab in the user's default web browser.

        After all the URLs have been opened, the script will exit.

        Note: This method requires the `webbrowser` and `sys` modules to be imported.
        """
        if "initiate" in self.user_choice and "morning" in self.user_choice and "protocol" in self.user_choice:
        
            self.say(f"Initiating morning protocol, {self.salutation}")
            
            
            morning_protocol_urls = ["https://www.linkedin.com", "https://app.beehiiv.com", "https://twitter.com/home", "https://chat.openai.com/"]
            def morning_web_protocol(urls):
                for url in urls:
                  webbrowser.open_new_tab(url)
            
            morning_web_protocol(morning_protocol_urls)
            self.supbox.append("a")
            
    def search_youtube(self): 
        """
        This method searches for a YouTube video based on user input and plays it.

        If the user has selected the 'youtube' and 'mode' options, the method prompts the user for a video to play
        and then searches YouTube for the video using the `play_youtube_video` function. The method then exits the program.

        Args:
            None

        Returns:
            None
        """
        if "youtube" in self.user_choice and "mode" in self.user_choice:
            self.say("What youtube video would you like to play?")
            video_choice = self.get_user_input()
            self.say("pulling it up now")
            play_youtube_video(query=video_choice)
            return False
        return True
            
        
    
    def art_mode(self):
        """Initiates an art mode and generates an image based on user input.

        Prompts the user for an image prompt and generates an image using the 
        generate_image() function. The program terminates after the image is generated.

        Raises:
            No exceptions are explicitly raised, but the generate_image() function 
            called within this method may raise exceptions if there are issues with 
            the image generation process.

        Returns:
            None
        """
        if "initiate" in self.user_choice and "art" in self.user_choice and "mode" in self.user_choice:
            self.say(f"Art mode initiated")
            self.say(f"what is it you would like me to make?")
            image_prompt = self.get_user_input()
            if image_prompt is not None:
                self.say(f"Generating a masterpiece for you")
                generate_image(prompt=image_prompt)       
                self.supbox.append("a")
                
         
    def check_stop(self):
        """
        Check if the user has requested to stop the conversation.

        If the user's choice is 'stop', 'bye', or 'no' (case-insensitive),
        this method prints a goodbye message and terminates the program using
        sys.exit().

        Returns: None
        """
        if self.user_choice.lower() in ["stop", "bye", "no"]:
            self.say("Okay, Goodbye")
            return False
        return True
            
            
            
    def stocks(self):
        if "stock" in self.user_choice and "chart" in self.user_choice: 
            self.say(f"What company would you like me to chart")
            stock_ticker = self.get_user_input()
            define_stock_gbt = f"Your task is to give me the stock ticker of the company i am talking about. Don't explain anything i just want the ticker that is it. nothing before or after it. THE COMPANY IS : {stock_ticker}"
            self.say(f"Generating chart now")
    
            stock_list = []
            stock_list.append({"role": "assistant", "content": f"{define_stock_gbt}"})
            stock_list.append({f"role": "user", "content": f"{self.user_choice}"})
        
            stock_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=stock_list
            )
            stock_list_data = stock_response.choices[0].message.content
            try:
             plot_stock_data(stock_list_data)
             self.supbox.append("a")
             return False
            except KeyError:
             self.say("sorry, there was an error in the process of generating the chart")
             self.supbox.append("a")
             return False
         
        if "stock" in self.user_choice and "terminal" in self.user_choice: 
             self.say("Initiating stock terminal now")
             subprocess.run(['start', 'cmd', '/c','D:\\OpenBB\\OpenBBTerminal.exe'],shell=True)
             return False
        return True
            
             
    def click_screen(self):
        if "click" in self.user_choice and "screen" in self.user_choice:
            self.say(f"Okay")
            x, y = 735, 483
            pyautogui.click(x, y)
            self.say(f"clicked the screen, {self.salutation}")
            self.supbox.append("a")
            
        if "click" in self.user_choice and "space" in self.user_choice and "bar" in self.user_choice: 
            self.say(f"Okay")
            pyautogui.press('space')
            pyautogui.sleep(1)
            pyautogui.keyUp('space') 
            self.supbox.append("a")
        if "shutdown" in self.user_choice and "computer" in self.user_choice:
            self.say("Are you sure you want to shut down the computer?")
            y_or_no = self.get_user_input().lower()
            
            if y_or_no == "yes":
              os.system("cmd /c shutdown /s /t 1")
            else:
              self.say("Didn't get a clear response. Cancelling shut down.")
              
            self.supbox.append("a")
            

    def check_object_detection(self):
     if "object" in self.user_choice and "detection" in self.user_choice:
         self.say("initiating object detection")
         object_detection()
         return False
     return True
    


            
            
    
            

            








        



