import os
import wave
import time
import threading
import tkinter as tk
from tkinter import Label
import pyaudio
import whisper
from datetime import datetime
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import config
import glob
# from transformers import pipeline
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# todo:
# recorder: Done
# transcribe: Done
# summary: Done

class MedicalTranscriber:
    def __init__(self):
        self.root= tk.Tk()
        self.root.geometry('400x200')
        # self.root.resizable(False, False)
        self.button = tk.Button(text="🎤", font=("Arial", 120, "bold"), command=self.click_handler) 
        self.button.pack()
        
        self.label = tk.Label(text="00:00:00")
        self.label.pack()
        
        # Create a label to display the status
        self.status_label = tk.Label(text="Status: Idle")
        self.status_label.pack()
        
        self.recording = False
        self.audio_file_path = ""
        
        #Transcribe:
        self.model = whisper.load_model('base.en')
        
        #Summary:
        self.anthropic = Anthropic()
        self.anthropic.api_key = config.CLAUDE_API_KEY
        
        self.root.mainloop()

    def click_handler(self):
        if self.recording:
            self.recording = False
            self.button.config(text="🎤", font=("Arial", 120, "bold"))
        else:
            self.recording = True
            self.button.config(text="🛑", font=("Arial", 120, "bold"))
            threading.Thread(target=self.record).start()
    
    def record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        frames = []
        self.status_label.config(text="Recording...")
        start_time= datetime.now()
        start = time.time()
        
        while self.recording:
            data = stream.read(1024)
            frames.append(data)
            
            passed = time.time() - start
            secs = passed % 60
            mins = passed // 60
            hours = mins // 60
            self.label.config(text=f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")

        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        end_time = datetime.now()
        audio_file_path = start_time.strftime("%y_%m_%d_%I:%M%p") + "_" + end_time.strftime("%I:%M%p") + ".wav"
                
        sound_file = wave.open(audio_file_path, "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()
        
        self.transcribe(audio_file_path)
        
        
    def claude_summary(self, text):
        
        instruction = """Below is a pulmonologist visit note. Summarize it in bullet points in two versions. 
            The first use medical terminology for pulmonary medicine for doctor notes. 
            The second one uses simple terms for patient summary. 
            Only summarize, do not add or remove information:\n"""
        
        completion = self.anthropic.completions.create(
            model="claude-2.0", #claude-2.0 #claude-instant-1.2
            max_tokens_to_sample=1000,
            temperature=0.1,
            prompt=f"""{HUMAN_PROMPT}{instruction}
            {text}
            {AI_PROMPT}:""",
        )
        return(completion.completion)
    
    
    def transcribe(self, audio_file_path):
        self.status_label.config(text="Transcribing...")
        result = self.model.transcribe(audio_file_path, fp16=False)
        transcribed_text = result['text']
        
        #Generate summary:
        if len(transcribed_text) > 100:
            self.status_label.config(text="Summarizing...")
            summary = self.claude_summary(transcribed_text)
        else:
            summary = "Transcribed text was too short to summarize."
            
        # Specify the folder for .txt files
        folder_path = os.path.join(os.getcwd(), 'transcribedText')
        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        txt_file_path = os.path.join(folder_path, os.path.splitext(os.path.basename(audio_file_path))[0] + '.txt')
        
        with open(txt_file_path, 'w') as f:
            # Write the summary at the top of the .txt file
            f.write("Summary:\n")
            f.write(summary)
            f.write("\n\nFull Transcription:\n")
            # Write the full transcription to the .txt file in blocks of 100 characters
            for i in range(0, len(transcribed_text), 100):
                f.write(transcribed_text[i:i+100])
                f.write("\n")
                        
        
        # # Find all .wav files in the current directory
        # wav_files = glob.glob('*.wav')
        # # Delete all .wav files
        # for wav_file in wav_files:
        #     os.remove(wav_file)
        
        # Update the GUI
        self.status_label.config(text="Finished.")
        

def main():
    MedicalTranscriber()

if __name__ == "__main__":
    main()