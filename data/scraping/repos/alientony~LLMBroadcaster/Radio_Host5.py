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

api_base = "http://localhost:5001/v1"
OPEN_WEATHER_MAP_API_KEY = ""
CITY = "London"  # Change to your desired city

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
        prompt=prompt,
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
    listbox_commands.insert(tk.END, "Queued: Comedy Skit")

def on_add_song():
    next_song = watcher.get_random_song()
    song_intro_prompt = f"Co-Host: Introduce the next song titled '{next_song[:-4]}' to the audience. Only say the name in a funny sentence. Announcer:"
    response = openai_api.Completion_create(
        model="gpt-4-32k",
        prompt=song_intro_prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7,
        api_base=api_base
    )
    song_intro = response["choices"][0]["text"]
    song_path = os.path.join(watcher.directory, next_song)
    song_queue.append(SongItem(song_intro, song_path))
    listbox_commands.insert(tk.END, f"Queued: {next_song}")

def play_from_queue():
    if not song_queue:
        return

    item = song_queue.pop(0)
    item.play()

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
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OPEN_WEATHER_MAP_API_KEY}&units=metric"
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
    
    prompt = f"Provide a fun weather forecast for today in {CITY}. The current temperature is {temperature}°C with {description}."
    
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

frame_right = tk.Frame(window)
frame_right.pack(side=tk.RIGHT, padx=20, pady=20)

listbox_commands = Listbox(frame_right)
listbox_commands.pack(pady=10)

if __name__ == "__main__":
    music_dir = "./music_folder"
    openai_api = OpenAI()
    watcher = MusicDirectoryWatcher(music_dir)

    window.mainloop()
