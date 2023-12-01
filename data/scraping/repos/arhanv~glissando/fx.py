from pedalboard import Compressor, Gain, Pedalboard, Chorus, Distortion, Reverb, PitchShift, Delay, Mix, Limiter
from pedalboard.io import AudioFile
import numpy as np
import openai
import re
import os

ranger = iter(range(1, 31))

openai.api_key = os.environ.get('KEY_PT')

#########
## Audio I/O ##
#########

def set_input(f_name = '/content/drive/MyDrive/Pedalboard Experiment/Dry Guitar.wav'):
  """
  Takes in an audio file as a file-path string, returns an AudioFile object.
  Use this method to select and sample the input file for processing.

    Parameters:
        f_name (str): Valid file path pointing to the "dry" audio.

    Returns:
        audio: AudioFile object that can be processed and exported via pedalboard.io
  """
  with AudioFile(f_name) as f:
    audio = f.read(f.frames)
  return audio
  
def write_output(input_audio, f_name, p_board, samplerate = 44100):
  """
  Takes in an AudioFile object, a valid path name, a Pedalboard object and
  (optionally) a desired sample rate. Applies the Pedalboard effects to
  the audio and exports it to the specified path name.

    Parameters:
        input_audio (AudioFile): Object representing dry audio for processing.
        f_name (str): File path specifying the output destination for processed or "wet" audio.
        p_board (Pedalboard): Object representing effects chain to be applied.
        samplerate (float): Sampling rate to be used for the export.

    Returns:
        effected (AudioFile): Object containing processed or "wet" audio.
  """
  effected = p_board(input_audio, samplerate)
  with AudioFile(f_name, 'w', samplerate, effected.shape[0]) as f:
    f.write(effected)
  return effected

#########
## Placeholder Genre Presets ##
#########

# Designing pedals for specific requests that can later be concatenated for a "composite" sound.
def chorus(x):
  return Chorus(rate_hz = 20, depth = 0.25, mix = x/100)

def drive(x):
  return Mix([Distortion(drive_db = 0.5 * x), Compressor(threshold_db=-1, ratio = 1.6)])

def delay(x):
  return Delay(delay_seconds=0.3, mix = x * 0.6 * 0.01)

def compressor(x):
  return Compressor(threshold_db=-6, ratio = x / 10)

def reverb(x):
  wet_l = min(0.5, x/100 + 0.3)
  return Reverb(room_size = 0.2, damping = 0.6, wet_level = wet_l, dry_level = 1 - wet_l)

pedals = {
  "Compressor": compressor,
  "Drive": drive,
  "Chorus": chorus,
  "Delay": delay,
  "Reverb": reverb
}

#########
## OOP Implementation of Generated Pedal ##
#########

system_prompt = "I am a user entering text prompts related to guitar tones. You need to return your best guess for a MAXIMUM of 3 parameters based on that prompt (setting the rest to 0%) in EXACTLY this format - Compressor: X%, Drive: X%, Chorus: X%, Delay: X%, Reverb: X% - where X is some percentage value between 1 and 99 if selected, and the rest are explicitly set to 0%. The goal is to select pedals and parameters that would possibly be the most accurate in replicating the sound demanded by the user."
class BoardGenerator:
  def __init__(self, input_text):
    self.input_text = input_text
    self.board = Pedalboard()
    self.weights = self.get_weights()
    self.board = self.make_board()

  def get_weights(self):
    user_token = self.input_text
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_token}
      ],
      max_tokens = 40
    )
    response = completion.choices[0].message["content"]
    self.gpt_response = response
    print("user token: " + str(user_token) + " || gpt response: " + str(response))
    percentage_pattern = re.compile(r"\d+%")
    percentage_values = percentage_pattern.findall(response)
    percentages = [int(value[:-1]) for value in percentage_values]
    return percentages

  def make_board(self):
    for i in range(5):
      if self.weights[i] > 0:
        self.board.append(list(pedals.values())[i](self.weights[i]))
    self.board.append(Limiter(threshold_db=-4))
    self.text = "Compressor: " + str(self.weights[0]) + "% Drive: " + str(self.weights[1]) + "% Chorus: " + str(self.weights[2]) + "% Delay: " + str(self.weights[3]) + "% Reverb: " + str(self.weights[4])
    return self.board
