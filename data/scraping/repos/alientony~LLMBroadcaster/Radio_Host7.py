import os
import random
import tkinter as tk
from tkinter import Listbox, Button, Checkbutton
import threading
from pydub import AudioSegment
from pydub.playback import play
import openai
from gtts import gTTS
import requests
import datetime
import configparser
import threading




# Load the configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Access the configuration settings in the code
api_base = config['DEFAULT']['api_base']
openai_api_key = config['OpenAI']['api_key']
openweather_api_key = config['OpenWeatherMap']['api_key']
city = config['OpenWeatherMap']['city']

song_intro_prompt = config['Prompts']['song_intro']
comedy_skit_prompt = config['Prompts']['comedy_skit']
weather_forecast_prompt = config['Prompts']['weather_forecast']



class LogBook:
    def __init__(self, filename="log_book.txt"):
        self.filename = filename

    def write_entry(self, entry_type, content):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filename, "a", encoding='utf-8') as f:
            f.write(f"{timestamp} - {entry_type}: {content}\n")


class OpenAI:
    def __init__(self):
        pass
    
    @staticmethod
    def Completion_create(model, prompt, max_tokens, n, stop, temperature, api_base):
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature,
            api_base=api_base
        )
        return response
        
log_book = LogBook()

class MusicDirectoryWatcher:
    def __init__(self, directory):
        self.directory = directory
        self.last_played = None
    
    def get_random_song(self):
        all_songs = [f for f in os.listdir(self.directory) if f.endswith(".mp3")]
        if len(all_songs) == 1:
            return all_songs[0]
        selectable_songs = [song for song in all_songs if song != self.last_played]
        selected_song = random.choice(selectable_songs)
        self.last_played = selected_song
        return selected_song

def generate_voice_announcement(text, lang="en"):
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_file = "audio_output.mp3"
    tts.save(audio_file)
    return audio_file

def play_audio_file(audio_file_path):
    audio_segment = AudioSegment.from_file(audio_file_path, format="mp3")
    play(audio_segment)

def generate_comedy_skit():
    prompt = "Create a short, funny comedy skit for the radio audience."
    response = openai_api.Completion_create(
        model="gpt-4-32k",
        prompt=comedy_skit_prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
        api_base=api_base
    )
    skit_text = response["choices"][0]["text"]
    skit_data = {
        "Voice Introduction Text": skit_text,
        "Type": "Comedy Skit"
    }
    song_queue.append(ComedySkitItem(skit_text))
    log_book.write_entry("Comedy Skit", skit_text)
    listbox_commands.insert(tk.END, "Queued: Comedy Skit")

def on_add_song():
    next_song = watcher.get_random_song()
    song_intro_prompt = f"Co-Host: Introduce the next song titled '{next_song[:-4]}' to the audience. Only say the name in a funny sentence. Announcer:"
    response = openai_api.Completion_create(
        model="gpt-4-32k",
        prompt=song_intro_prompt.format(song_name=next_song[:-4]),
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
        api_base=api_base
    )
    song_intro = response["choices"][0]["text"]
    song_path = os.path.join(watcher.directory, next_song)
    song_queue.append(SongItem(song_intro, song_path))
    log_book.write_entry("Song Intro", song_intro)
    listbox_commands.insert(tk.END, f"Queued: {next_song}")

is_playing = False
play_lock = threading.Lock()

def play_from_queue():
    global is_playing
    
    with play_lock:
        if is_playing:
            print("Already playing, please wait...")
            return
        is_playing = True
    
    if not song_queue:
        with play_lock:
            is_playing = False
        return
    
    item = song_queue.pop(0)
    
    try:
        item.play()
    finally:
        with play_lock:
            is_playing = False

        listbox_commands.delete(0)

        # Check if auto_play_next is True and play next item if available
        if auto_play_next.get() and song_queue:
            window.after(1000, threaded_play_from_queue)  # Delay of 1 second before next play

def threaded_play_from_queue():
    threading.Thread(target=play_from_queue).start()

def move_up():
    index = listbox_commands.curselection()
    if not index:
        return
    index = index[0]
    if index == 0:
        return
    song_queue[index], song_queue[index-1] = song_queue[index-1], song_queue[index]
    listbox_commands.insert(index-1, listbox_commands.get(index))
    listbox_commands.delete(index+1)
    listbox_commands.selection_set(index-1)

def move_down():
    index = listbox_commands.curselection()
    if not index:
        return
    index = index[0]
    if index == len(song_queue) - 1:
        return
    song_queue[index], song_queue[index+1] = song_queue[index+1], song_queue[index]
    listbox_commands.insert(index+2, listbox_commands.get(index))
    listbox_commands.delete(index)
    listbox_commands.selection_set(index+1)

def toggle_auto_play():
    if auto_play_next.get():
        btn_toggle_auto_play.config(text="Auto-Play: ON")
    else:
        btn_toggle_auto_play.config(text="Auto-Play: OFF")




def fetch_weather():
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={openweather_api_key}&units=metric"
    response = requests.get(base_url)
    data = response.json()

    if 'main' not in data:
        print(f"Error fetching weather data: {data}")
        return None, None

    temperature = data["main"]["temp"]
    description = data["weather"][0]["description"]
    return temperature, description


def on_generate_weather_forecast():
    temperature, description = fetch_weather()
    if temperature is None or description is None:
        print("Failed to fetch weather data.")
        return

    prompt = weather_forecast_prompt.format(city=city, temperature=temperature, description=description)
    response = openai_api.Completion_create(
        model="gpt-4-32k",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
        api_base=api_base
    )

    forecast_text = response["choices"][0]["text"]
    song_queue.append(WeatherForecastItem(forecast_text))
    listbox_commands.insert(tk.END, "Queued: Weather Forecast")
    log_book.write_entry("Weather Forecast", forecast_text)

class QueueItem:
    def __init__(self, voice_intro_text):
        self.voice_intro_text = voice_intro_text

    def play(self):
        voice_intro_file = generate_voice_announcement(self.voice_intro_text)
        play_audio_file(voice_intro_file)

class SongItem(QueueItem):
    def __init__(self, voice_intro_text, song_file):
        super().__init__(voice_intro_text)
        self.song_file = song_file

    def play(self):
        super().play()
        play_audio_file(self.song_file)

class ComedySkitItem(QueueItem):
    pass

class WeatherForecastItem(QueueItem):
    pass
    
def open_settings():
    settings_window = tk.Toplevel(window)
    settings_window.title("Settings")

    tk.Label(settings_window, text="API Base:").pack(pady=5)
    api_base_entry = tk.Entry(settings_window, width=40)
    api_base_entry.insert(0, api_base)
    api_base_entry.pack(pady=5)


    tk.Label(settings_window, text="OpenAI API Key:").pack(pady=5)
    openai_api_key_entry = tk.Entry(settings_window, width=40)
    openai_api_key_entry.insert(0, openai_api_key)
    openai_api_key_entry.pack(pady=5)

    tk.Label(settings_window, text="OpenWeatherMap API Key:").pack(pady=5)
    openweather_api_key_entry = tk.Entry(settings_window, width=40)
    openweather_api_key_entry.insert(0, openweather_api_key)
    openweather_api_key_entry.pack(pady=5)

    tk.Label(settings_window, text="City:").pack(pady=5)
    city_entry = tk.Entry(settings_window, width=40)
    city_entry.insert(0, city)
    city_entry.pack(pady=5)
    
    tk.Label(settings_window, text="Song Intro Prompt:").pack(pady=5)
    song_intro_prompt_entry = tk.Entry(settings_window, width=40)
    song_intro_prompt_entry.insert(0, song_intro_prompt)
    song_intro_prompt_entry.pack(pady=5)

    tk.Label(settings_window, text="Comedy Skit Prompt:").pack(pady=5)
    comedy_skit_prompt_entry = tk.Entry(settings_window, width=40)
    comedy_skit_prompt_entry.insert(0, comedy_skit_prompt)
    comedy_skit_prompt_entry.pack(pady=5)

    tk.Label(settings_window, text="Weather Forecast Prompt:").pack(pady=5)
    weather_forecast_prompt_entry = tk.Entry(settings_window, width=40)
    weather_forecast_prompt_entry.insert(0, weather_forecast_prompt)
    weather_forecast_prompt_entry.pack(pady=5)
    


    def save_settings():
        config.set('OpenAI', 'api_key', openai_api_key_entry.get())
        config.set('DEFAULT', 'api_base', api_base_entry.get())
        config.set('OpenWeatherMap', 'api_key', openweather_api_key_entry.get())
        config.set('OpenWeatherMap', 'city', city_entry.get())
        config.set('Prompts', 'song_intro', song_intro_prompt_entry.get())
        config.set('Prompts', 'comedy_skit', comedy_skit_prompt_entry.get())
        config.set('Prompts', 'weather_forecast', weather_forecast_prompt_entry.get())
        with open('config.ini', 'w') as configfile:
            config.write(configfile)

        # Update the global variables
        global api_base, openai_api_key, openweather_api_key, city
        global song_intro_prompt, comedy_skit_prompt, weather_forecast_prompt
        api_base = api_base_entry.get()
        openai_api_key = openai_api_key_entry.get()
        openweather_api_key = openweather_api_key_entry.get()
        city = city_entry.get()
        song_intro_prompt = song_intro_prompt_entry.get()
        comedy_skit_prompt = comedy_skit_prompt_entry.get()
        weather_forecast_prompt = weather_forecast_prompt_entry.get()

        settings_window.destroy()

    tk.Button(settings_window, text="Save", command=save_settings).pack(pady=20)
    

window = tk.Tk()
window.title("Radio Host")

song_queue = []
auto_play_next = tk.BooleanVar(value=False)

frame_left = tk.Frame(window)
frame_left.pack(side=tk.LEFT, padx=20, pady=20)

btn_add_song = Button(frame_left, text="Add Song", command=on_add_song)
btn_add_song.pack(pady=10)

btn_generate_skit = Button(frame_left, text="Generate Comedy Skit", command=generate_comedy_skit)
btn_generate_skit.pack(pady=10)

btn_play_next = Button(frame_left, text="Play Next", command=threaded_play_from_queue)
btn_play_next.pack(pady=10)

btn_move_up = Button(frame_left, text="Move Up", command=move_up)
btn_move_up.pack(pady=10)

btn_move_down = Button(frame_left, text="Move Down", command=move_down)
btn_move_down.pack(pady=10)

btn_toggle_auto_play = Checkbutton(frame_left, text="Auto-Play: OFF", variable=auto_play_next, command=toggle_auto_play)
btn_toggle_auto_play.pack(pady=10)

btn_generate_weather = Button(frame_left, text="Generate Weather Forecast", command=on_generate_weather_forecast)
btn_generate_weather.pack(pady=10)

btn_settings = Button(frame_left, text="Settings", command=open_settings)
btn_settings.pack(pady=10)

frame_right = tk.Frame(window)
frame_right.pack(side=tk.RIGHT, padx=20, pady=20)

listbox_commands = Listbox(frame_right)
listbox_commands.pack(pady=10)

if __name__ == "__main__":
    music_dir = "./music_folder"
    openai_api = OpenAI()
    openai.api_key = openai_api_key  # Setting the OpenAI API key
    watcher = MusicDirectoryWatcher(music_dir)

    window.mainloop()

