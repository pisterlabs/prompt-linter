import tkinter as tk
from tkinter import simpledialog, messagebox, Listbox
import schedule
import time
import random
import multion
import openai
import os
import threading
from langchain.agents.agent_toolkits import MultionToolkit
from langchain.document_loaders import ArxivLoader
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType

openai.api_key = os.environ.get('OPENAI_API_KEY')
llm = OpenAI(temperature=0)

class SongScheduler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YouTube Song Scheduler")
        self.geometry("500x400")
        
        # Time Entry
        self.time_label = tk.Label(self, text="Set Time (24hr format HH:MM):")
        self.time_label.pack(pady=10)
        self.time_entry = tk.Entry(self, width=30)
        self.time_entry.pack(pady=10)
        
        # Song Listbox
        self.song_label = tk.Label(self, text="Songs List:")
        self.song_label.pack(pady=5)
        self.song_listbox = Listbox(self, height=10, width=40)
        self.song_listbox.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Add and Remove Buttons
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=10)
        self.add_button = tk.Button(self.button_frame, text="Add Song", command=self.add_song)
        self.add_button.grid(row=0, column=0, padx=10)
        self.remove_button = tk.Button(self.button_frame, text="Remove Song", command=self.remove_song)
        self.remove_button.grid(row=0, column=1, padx=10)
        
        # Save Alarm Button
        self.save_button = tk.Button(self, text="Save Alarm", command=self.save_alarm)
        self.save_button.pack(pady=20)
        
    def add_song(self):
        song = simpledialog.askstring("Input", "Enter the song name:")
        if song:
            self.song_listbox.insert(tk.END, song)
            
    def remove_song(self):
        self.song_listbox.delete(tk.ACTIVE)
        
    def save_alarm(self):
        set_time = self.time_entry.get()
        songs = self.song_listbox.get(0, tk.END)
        
        if not songs:
            messagebox.showwarning("Warning", "Please add at least one song.")
            return
        if not set_time or len(set_time.split(":")) != 2:
            messagebox.showwarning("Warning", "Please set a valid time in HH:MM format.")
            return
        
        # Schedule the task based on the time set in the GUI
        schedule.every().day.at(set_time).do(self.play_song)
        
        # Confirmation message
        messagebox.showinfo("Info", f"Alarm set for {set_time} with {len(songs)} songs.")
        
    def play_song(self):
        # Get all songs from the listbox
        songs = self.song_listbox.get(0, tk.END)
        if not songs:
            return
        song_to_play = random.choice(songs)
        
        # Play song using multion
        multion.login()
        toolkit = MultionToolkit()
        tools = toolkit.get_tools()
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        response = multion.new_session({
            "input": f"search for {song_to_play} on YouTube and play the first result",
            "url": "https://www.youtube.com"
        })
        # Handle the response, check for errors, etc.

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Start the scheduling loop in a separate thread
scheduler_thread = threading.Thread(target=run_schedule)
scheduler_thread.start()

# Start the application
app = SongScheduler()
app.mainloop()
