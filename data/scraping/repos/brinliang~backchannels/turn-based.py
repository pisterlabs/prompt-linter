import speech_recognition as sr
import pyttsx3
import openai
import time
import config

openai.api_key = config.openai_key # set api key

r = sr.Recognizer() # recognizer instance
m = sr.Microphone() # audio input instance
engine = pyttsx3.init() # audio output instance

# prompt for gpt3 input
prompt = 'Respond with a verbal backchannel as if you are actively listening to someone say "{}"'

# set audio input threshold and calibrate
r.energy_threshold = 1000
r.dynamic_energy_threshold = False
with m:
  r.adjust_for_ambient_noise(m)

# set pause between phrases threshold, default is 0.8 seconds
# r.pause_threshold = 0.8

while True:
  # listen for audio input
  start = time.time()
  print('listening...')
  with m:
    audio = r.listen(m)
  print('audio received after {0:.4f} seconds'.format(time.time() - start))

  # transcribe
  try: 
    lap = time.time()
    print('\ntranscribing...')
    transcript = r.recognize_google(audio)
    print(transcript)
    print('transcribed in {0:.4f} seconds'.format(time.time() - lap))
  except:
    print('failed to transcribe\nplease try again')
    continue

  # generate response
  try:
    lap = time.time()
    print('\ngenerating response...')
    gpt3_input = prompt.format(transcript)
    gpt3_output = openai.Completion.create(
      model='text-davinci-003',
      prompt=gpt3_input,
      max_tokens=256,
    )
    print(gpt3_output)
    response = gpt3_output['choices'][0]['text'].strip()
    print('generated in {0:.4f} seconds'.format(time.time() - lap))
  except:
    print('failed to generate response\ncheck to make sure your api key is valid')
    continue

  # play response
  lap = time.time()
  print('\nplaying response...')
  print(response)
  engine.say(response)
  print('played in {0:.4f} seconds'.format(time.time() - lap))
  print('finished in {0:.4f} seconds'.format(time.time() - start))
  engine.runAndWait()
  engine.stop()