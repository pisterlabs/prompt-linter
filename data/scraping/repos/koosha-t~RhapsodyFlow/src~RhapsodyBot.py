import os
from dotenv import load_dotenv
import openai
from rhapsody_agent import find_violating_measures, count_measures
from MusicBluePrint import MusicBluePrint

# Reading openai org and key from env variables
env_path = os.path.join(os.path.dirname(__file__),f"../.env")
load_dotenv(dotenv_path=env_path)
openai.organization = os.getenv('openai_org')
openai.api_key = os.getenv('openai_key')


class RhapsodyComposerBot:

  def __init__(self, blue_print: MusicBluePrint, system_message, model='gpt-4', temperature = 0.8):
    self.system_message = system_message
    self.model = model
    self.temperature = temperature
    self.blue_print = blue_print
    self.initial_user_prompt = self._get_initial_user_prompt()

  def _get_initial_user_prompt(self):
    prompt = f"Compose a '{self.blue_print.genre}' '{self.blue_print.mood}' Piano music sheet in the '{self.blue_print.scale}' scale and in " +\
    f"'{self.blue_print.time_signature}' time signiture, inspired by {self.blue_print.inspired_by}, using  MeloCode. Please ensure that the song has a beautiful harmony. " + \
    f"Important: Please answer in MeloCode only, do NOT include any English sentences in your response."
    return prompt

  def _get_completion_from_messages(self, messages):
    response = openai.ChatCompletion.create(
        model=self.model,
        messages=messages,
        temperature=self.temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]
  

  def _iterate_till_all_measures_are_valid(self, messages):
    flag = True
    while flag:
      score = self._get_completion_from_messages(messages=messages)
      print(f"*** GPT4 ****\n\n {score}\n\n\n")
      v_measures = find_violating_measures(time_signature=self.blue_print.time_signature, melocode=score)
      if len(v_measures) > 0:
        agent_message = f"Here is the score we have:\n {score}\n\n. " + \
            f"The following measures in the above score violate the {self.blue_print.time_signature} time signiture:\n {v_measures}\n please fix them." + \
            f"Write the whole song again with the fixed measures. Write the MeloCode score only, don't include any English sentences in your response."
        print(f"*** RhabsodyBot ***:\n\n{agent_message}\n\n\n")
        messages = messages = [
          {'role': 'system', 'content':system_message},
          {'role': 'user', 'content': agent_message},
        ]
      else:
        flag=False
    return score

  
  def compose_music(self):
    messages = [
      {'role': 'system', 'content':self.system_message},
      {'role': 'user', 'content': self.initial_user_prompt},
    ] 
    print(f"**** SYSTEM MESSAGE *****\n\n{self.system_message}\n\n\n")
    print(f"*** Rhabsody Bot ****\n\n{self.initial_user_prompt}\n\n\n")

    score = self._iterate_till_all_measures_are_valid(messages)

    improvement_request = f"Make the following music more advanced by giving it more variation and more depth." +\
      f" Use more chords and notes with lower durations as it's supposed to be played by professional pianists. \n {score}\n Answer in MeloCode only, do NOT include any English sentences in your response"
    
    print(f"*** Rhapsody Bot ***\n\n {improvement_request}\n\n\n")

    messages = [
      {'role':'system','content':self.system_message},
      {'role':'user', 'content':improvement_request}
    ]

    score = self._iterate_till_all_measures_are_valid(messages)

    return score




if __name__ == "__main__":

  system_message=  """
  You are a professional classical music composer who can write musing in MeloCode notation system; for Piano. 
  The following is the MeloCode's description, delimited by '***': 

  ***
  - Pitch:
      ```
      - Uppercase letters for natural notes (C, D, E, F, G, A, B)
      - '#' for sharps and 'b' for flats (C#, Db, D#, Eb, F#, Gb, G#, Ab, A#, Bb)
      - numbers to indicate the octave of the notes (C4: middle C, A3: A below middle C)
      ```
  - Duration: 
    ```
    Numbers as fractions for note duration, placing them before the pitch notation:
      - 1: whole note (1C4: whole note C in the 4th octave)
      - 1/2: half note (1/2D4: half note D in the 4th octave)
      - 1/4: quarter note (1/4E4: quarter note E in the 4th octave)
      - 1/8: eighth note (1/8F4: eighth note F in the 4th octave)
      - 1/16: sixteenth note (1/16G4: sixteenth note G in the 4th octave)
      - 1/32: thirty-second note (1/32A4: thirty-second note A in the 4th octave)
    ```
  - Rest: 
    ```
    Uppercase 'R' followed by the duration number, placing them before or after a note as needed:
      - R1: whole rest (1/4C4 R1 1/4C4: quarter note C, whole rest, quarter note C)
      - R1/2: half rest (1/2D4 R1/2 1/4D4: half note D, half rest, quarter note D)
      - R1/4: quarter rest (1/8E4 R1/4 1/8E4: eighth note E, quarter rest, eighth note E)
      - R1/8: eighth rest (1/4F4 R1/8 1/16F4: quarter note F, eighth rest, sixteenth note F)
      - R1/16: sixteenth rest (1/16G4 R1/16 1/32G4: sixteenth note G, sixteenth rest, thirty-second note G)
    ```
  -  Playing with both hands: 
    ```Use two ampersands && to separate the right hand (treble clef) and left hand (bass clef) notations (i.e. anything before && is for the treble clef - right hand - and anything after && is for bass clef - the left hand.)
      - Example:  | 1/4C4 1/4E4 1/4G4 1/4C5 && 1/4C2 1/4G2 1/4C3 1/4G3 |
    ```
  - Bar lines and time signature: 
  ```
  '|' for bar lines and a tuple for the time signature.
      - Time signature is written at the begenning of the piece ONLY. Example:
          - (4/4): time signature (4 beats per measure, with a quarter note receiving one beat)
      - The following example features a 4/4 time signature and three measures separated by bar lines for both hands:
          - (4/4) | 1/4C4 1/4E4 1/4G4 1/4C5 && 1/4C2 1/4G2 1/4C3 1/4G3 | R1/4 1/8D4 1/8F4 1/4A4 1/4D5 && R1/2 1/4E2 1/4G2 1/4E3 | R1/2 1/4E4 1/4G4 1/4E5 && R1/2 1/4G2 1/4B2 1/4G3 |
  ```
  - Dynamics:
    ```
    angle brackets with the first letter of the dynamic marking.
      - `<p>`: piano (soft)
      - `<f>`: forte (loud)
      - `<m>`: mezzo (medium)
      - Example: (4/4) | <p> 1/4C4 1/4E4 <m> 1/4G4 <f> 1/4C5 && <p> 1/4C2 1/4G2 <m> 1/4C3 <f> 1/4G3 | R1/4 <p> 1/8D4 1/8F4 <f> 1/4A4 1/4D5 && R1/2 <p> 1/4E2 1/4G2 <f> 1/4E3 | R1/2 <m> 1/4E4 1/4G4 1/4E5 && R1/2 <m> 1/4G2 1/4B2 1/4G3 |
    ```
  - Chord: 
  ```
  Use square brackets [ and ] to enclose pitches that should be played together as a chord, separating each pitch with a comma.
      - Example: (4/4) | 1/4[C4,E4,G4] R1/2 && 1/4[C2,G2,C3] R1/2 |: a quarter note C major chord in the 4th octave in the treble clef, and a quarter note C major chord in the 2nd octave in the bass clef.
  ```
  ***

  """

  music_blue_print = MusicBluePrint(scale='E-Minor', time_signature='3/4', tempo='120', genre='classic', mood="love", inspired_by="Yann Tiersen" )
  rhabsody_bot = RhapsodyComposerBot(blue_print=music_blue_print, system_message=system_message)
  rhabsody_bot.compose_music()





  