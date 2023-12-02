import datetime
import multiprocessing
import time
import webbrowser
import openai
import pyttsx3
import requests
import speech_recognition as sr
import smtplib
import os
import keyboard
import subprocess
import tkinter as tk
from tkinter import simpledialog
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from PIL import Image, ImageTk
from newsapi import NewsApiClient
from expense_manager import create_expense_manager
from maze_solver import create_maze_solver
from tic_tac_toe import TicTacToe
from note_taker import create_note_taker
from custom_commands_manager import CustomCommandsManager

load_dotenv()

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
volume = engine.getProperty('volume')
engine.setProperty('volume', 100.0)

openai.api_key = os.getenv('openai_key')
gmail_pass = os.getenv('gmail_pass_env')
gmail_id = os.getenv('gmail_id_env')

# Function to speak text
def _speak(text):
    engine.say(text)
    engine.runAndWait()

def speak(text):
    p = multiprocessing.Process(target=_speak, args=(text,))
    p.start()
    while p.is_alive():
        if keyboard.is_pressed('esc'):
            p.terminate()
        else:
            continue
    p.join()

# Function to get user input through speech
def get_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=10)

        try:
            command = recognizer.recognize_google(audio).lower()
            print("You said:", command)
            return command
        except sr.UnknownValueError:
            speak("Sorry, I couldn't understand. Please write the prompt.")
            command = input("Type the prompt: ")
            return command
        except sr.RequestError as e:
            speak("Error connecting to Google API: {0}".format(e))
            return ""

# Function to send an email
def send_email(receiver, subject, body):
    # Configure your email server here
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(gmail_id, gmail_pass)

    message = f"Subject: {subject}\n\n{body}"
    server.sendmail(gmail_id, receiver, message)

    server.quit()

# Function to get GPT response
def get_gpt_response(command):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=command,
        max_tokens=50
    )

    return response.choices[0].text.strip()

# Function to set a reminder
def set_reminder():
    print(datetime.now().time())
    speak("What would you like to be reminded about?")
    reminder_text = get_audio()

    if reminder_text:
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Create a simple dialog for picking time
        time_str = simpledialog.askstring("Set Reminder", "When should I remind you? (HH:MM:SS)")
        root.destroy()  # Close the dialog window

        if time_str:
            try:
                reminder_time = datetime.strptime(time_str, "%H:%M:%S").time()
                now = datetime.now().time()
                if reminder_time > now:
                    # Calculate the time difference
                    time_difference = datetime.combine(datetime.today(), reminder_time) - datetime.combine(
                        datetime.today(), now)

                    # Schedule the reminder
                    seconds_until_reminder = time_difference.total_seconds()
                    time.sleep(seconds_until_reminder)
                    speak(f"Reminder: {reminder_text}")
                    print(f"Reminder: {reminder_text}")
                else:
                    speak("Invalid reminder time. Please provide a future time.")
                    print("Invalid reminder time. Please provide a future time.")
            except ValueError:
                speak("Invalid time format. Please provide the time in HH:MM:SS format.")
                print("Invalid time format. Please provide the time in HH:MM:SS format.")

def get_lat_lon(city):
    geolocator = Nominatim(user_agent="your_app_name")
    location = geolocator.geocode(city)

    if location:
        return location.latitude, location.longitude
    else:
        return None, None

# Function to get weather information
def get_weather(city):
    api_key = os.getenv('weather_api_key')
    lat, lon = get_lat_lon(city)

    if lat is not None and lon is not None:
        weather_api_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"

        try:
            response = requests.get(weather_api_url)
            data = response.json()

            if response.status_code == 200:
                temperature = round((data["main"]["temp"]) - 273.15, 2)
                weather_description = data["weather"][0]["description"]
                print(temperature, weather_description)
                speak(
                    f"The current temperature in {city} is {temperature} Celsius, and the weather is {weather_description}.")
            else:
                speak("Sorry, I couldn't retrieve weather information for that city.")
        except Exception as e:
            print("Error fetching weather data:", e)
            speak("Sorry, there was an error fetching weather information.")


# function to get news updates
def get_news_updates():
    api_key = os.getenv('news_api_key')
    newsapi = NewsApiClient(api_key=api_key)
    news_api_url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"

    try:
        response = newsapi.get_top_headlines()
        if response is not None:
            data = response

            if 'articles' in data:
                articles = data['articles']
                for idx, article in enumerate(articles[:5]):
                    title = article['title']
                    description = article['description']
                    print(f"News {idx + 1}: {title}. {description}")
                    speak(f"News {idx + 1}: {title}. {description}")
            else:
                print(response)
                speak("Sorry, I couldn't retrieve news updates.")
    except Exception as e:
        print("Error fetching news data:", e)
        speak("Sorry, there was an error fetching news updates.")

def on_expense_manager_click():
    create_expense_manager()

def on_maze_solver_click():
    create_maze_solver()

# Function to open Note Taker
def on_note_taker_click():
    create_note_taker() 

# Function to handle custom command button click
def on_custom_command_click():
    custom_commands_manager = CustomCommandsManager()
    # Run the manager and fetch the updated command_program_dict
    custom_commands_manager.get_command_program_dict()

def start_tic_tac_toe_game():
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()

# Function to execute commands
def execute_command(command):
    if "hello" in command:
        speak('Hello! How can I assist you today?')

    elif "time" in command:
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        speak("The current time is " + current_time)

    elif "open Google" in command:
        webbrowser.open("https://www.google.com")

    elif "tell me the weather" in command:
        speak("Sure, I can provide weather information. What city are you interested in?")
        city = get_audio()
        if city:
            get_weather(city)

    elif "search" in command:
        speak("What would you like me to search for?")
        search_query = get_audio()
        webbrowser.open(f"https://www.google.com/search?q={search_query}")

    elif "send email" in command:
        speak("Who is the recipient?")
        recipient = simpledialog.askstring("Recipient", "Enter the recipient's email:")
        if recipient:
            speak("What is the subject of the email?")
            subject = simpledialog.askstring("Subject", "Enter the email subject:")
            speak("What should I say in the email?")
            body = simpledialog.askstring("Body", "Enter the email body:")
            send_email(recipient, subject, body)
            speak("Email sent successfully.")

    elif "set a reminder" in command:
        set_reminder()

    elif "note" in command:
        create_note_taker()

    elif "play music" in command:
        music_dir = "C:\\Users\\YourUsername\\Music"
        songs = os.listdir(music_dir)
        os.startfile(os.path.join(music_dir, songs[0]))

    elif "news updates" in command:
        get_news_updates()

    elif "exit" in command:
        speak("Goodbye!")
        root.destroy()
        exit()
    else:
        # Check if it's a custom command
        command_program_dict = CustomCommandsManager().command_program_dict
        if command in command_program_dict:
            program = command_program_dict[command]
            subprocess.Popen([program], shell=True)

        else:
            try:
                speak(f"Do you want information about {command}?")
                response = get_audio()

                if response and "yes" in response:
                    gpt_response = get_gpt_response(command)
                    print(gpt_response)
                    speak(gpt_response)
                else:
                    speak("Alright, let me know if there's anything else I can help you with.")
            except Exception as e:
                print("Error:", e)
                speak("Sorry, there was an error processing your request.")


# Function to handle button click
def on_submit_click():
    user_input = entry.get()
    entry.delete(0, tk.END)  # Clear the entry field
    execute_command(user_input)


# Function to handle microphone button click
def on_mic_click():
    user_input = get_audio()
    if user_input:
        entry.insert(tk.END, user_input)
    execute_command(user_input)


# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    root.title("AssistantX")
    root.configure(bg='#333333')  # Set dark background color

    main_frame = tk.Frame(root, padx=20, pady=20, bg='#333333')  # Set dark background color
    main_frame.pack()

    title_label = tk.Label(main_frame, text="AssistantX - Your Personal Assistant", font=("Helvetica", 20), bg='#333333', fg='white')  # Set text color to white
    title_label.pack(pady=10)

    label = tk.Label(root, text="AssistantX is ready. How can I assist you?", font=("Helvetica", 14), bg='#333333', fg='white')  # Set text color to white
    label.pack(pady=10)

    entry = tk.Entry(root, width=50, font=("Helvetica", 12))
    entry.pack(pady=10)

    submit_button = tk.Button(root, text="Submit", command=on_submit_click, bg="#4CAF50", fg="white", font=("Helvetica", 12))
    submit_button.pack(pady=10)

    # Stylish microphone button
    mic_icon_path = r"src\images\microphone.png"
    mic_icon = Image.open(mic_icon_path)
    mic_icon = mic_icon.resize((30, 30))
    mic_icon = ImageTk.PhotoImage(mic_icon)
    mic_button = tk.Button(root, image=mic_icon, command=on_mic_click, bd=0, highlightthickness=0, bg='#333333')  # Set dark background color
    mic_button.image = mic_icon
    mic_button.pack(pady=10)

    # Stylish action buttons
    buttons_frame = tk.Frame(root, bg='#333333')  # Set dark background color
    buttons_frame.pack(pady=10)

    action_buttons = [
        ("Expense Manager", on_expense_manager_click),
        ("Maze Solver", on_maze_solver_click),
        ("Play Tic Tac Toe", start_tic_tac_toe_game),
        ("Open Note Taker", on_note_taker_click),
        ("Manage Custom Commands", on_custom_command_click)
    ]

    for text, command in action_buttons:
        action_button = tk.Button(buttons_frame, text=text, command=command, font=("Helvetica", 12), bg='#4CAF50', fg='white')  # Set dark background color and white text
        action_button.pack(side=tk.LEFT, padx=10)

    root.mainloop()