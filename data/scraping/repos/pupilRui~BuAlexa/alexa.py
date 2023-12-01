###############################################################################################################
# Import necessary libraries and modules like AudioSegment, play, speech_recognition, whisper, etc.
###############################################################################################################
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
import whisper
import queue
import os
import threading
import threading
import torch
import numpy as np
import re
from gtts import gTTS
import openai
import click

###############################################################################################################
# Call the init_api() function to initialize API credentials using data from a ".env" file.
###############################################################################################################
def init_api():
    with open(".env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value

    openai.api_key = os.environ.get("API_KEY")
    openai.organization = os.environ.get("ORG_ID")



###############################################################################################################
# Let’s write the main function that calls the 3 functions each in a separate thread: Thread 1, 2, 3
# - read from arguments
###############################################################################################################
@click.command()
@click.option("--model", default="base", help="Model to use", type=click.Choice(["tiny", "base", "small", 
              "medium", "large"]))
@click.option("--english", default=False, help="Whether to use the English model", is_flag=True, type=bool)
@click.option("--energy", default=300, help="Energy level for the mic to detect", type=int)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--dynamic_energy", default=False, is_flag=True, help="Flag to enable dynamic energy", type=bool)
@click.option("--wake_word", default="hey computer", help="Wake word to listen for", type=str)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)

###############################################################################################################
# Define the main() function that:
# - Adjusts the model name if English is selected and the model is not "large".
# - Loads the audio model using whisper.load_model().
# - Creates queues for audio and result data.
# - Starts threads to run record_audio, transcribe_forever, and reply functions concurrently.
# - Prints results from the result queue indefinitely.
###############################################################################################################
def main(model, english, energy, pause, dynamic_energy, wake_word, verbose):

    ###########################################################################################################
    # There are no English models for "large", so if the chosen model is not "large" and the "english" flag 
    # is set, the model name is adjusted accordingly.
    # - There are no English models for "large."
    ###########################################################################################################
    if model != "large" and english:
        model = model + ".en"

    ###########################################################################################################
    # - Load Audio Model
    #   + The appropriate model for speech recognition is loaded using the specified options.
    ###########################################################################################################
    audio_model = whisper.load_model(model)
    audio_queue = queue.Queue()
    result_queue = queue.Queue()

    ###########################################################################################################
    # Thread 1
    #
    # - Start Recording Thread
    #   + A thread is started to run the record_audio function with the provided arguments
    # - Define the record_audio() function that records audio using the speech recognition library. 
    #   + Audio data is processed and added to the audio queue.
    ###########################################################################################################
    threading.Thread(target=record_audio, args=(audio_queue, energy, pause, dynamic_energy,)).start()

    ###########################################################################################################
    # Thread 2
    #
    # - Start Transcription Thread
    #   + A thread is started to run the transcribe_forever function with the provided arguments
    # - Define the transcribe_forever() function that transcribes audio using the Whisper model. 
    #   + Transcribed text is processed and added to the result queue.

    ###########################################################################################################
    threading.Thread(target=transcribe_forever, args=(audio_queue, result_queue, audio_model, english, wake_word, 
                     verbose,)).start()

    ###########################################################################################################
    # Thread 3
    #
    # - Start Reply Thread
    #   + A thread is started to run the reply function with the provided arguments
    # - Define the reply() function that generates a text response using the OpenAI API and converts it 
    #   + to audio using gTTS. The audio response is played and the temporary file is removed.
    ###########################################################################################################
    threading.Thread(target=reply, args=(result_queue, verbose,)).start()

    ###########################################################################################################
    # - Infinite Loop
    #   + An infinite loop prints response text obtained from the result_queue.
    ###########################################################################################################
    while True:
        print(result_queue.get())


###############################################################################################################
# - The function above record_audio records audio from a microphone 
#  and saves it to a queue for further processing.
# - The function takes four arguments:
#   + audio_queue
#     - A queue object where the recorded audio will be saved.
#   + energy
#     - An initial energy threshold for detecting speech.
#   + pause
#     - A pause threshold for detecting when speech has ended.
#   + dynamic_energy
#     - A Boolean indicating whether the energy threshold should be 
#       adjusted dynamically based on the surrounding environment.
###############################################################################################################
def record_audio(audio_queue, energy, pause, dynamic_energy):
    ################################################################
    # Create a speech recognizer and set energy and pause thresholds
    ################################################################
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = energy
    recognizer.pause_threshold = pause
    recognizer.dynamic_energy_threshold = dynamic_energy

    ################################################################
    # A microphone is initialized with a 16 kHz sample rate
    # - The loop continues and the function keeps recording audio 
    #   until it is interrupted.
    ################################################################
    with sr.Microphone(sample_rate=16000) as source:
        print("Listening...")
        i = 0

        ############################################################
        # The function enters an infinite loop.
        ############################################################
        while True:
            ########################################################
            # # Get and save audio to a WAV file
            # - During each iteration of the loop, the listen() 
            #   method of the recognizer object is called to 
            #   record audio from the microphone. 
            ########################################################
            audio = recognizer.listen(source)

            ########################################################
            # Convert audio to Torch tensor
            # - The recorded audio is converted into a 
            #   PyTorch tensor, normalized to a range of [-1, 1], 
            #   and then saved to the audio_queue. 
            ########################################################
            torch_audio = torch.from_numpy(
                np.frombuffer(audio.get_raw_data(), np.int16)
                .flatten()
                .astype(np.float32) / 32768.0
            )

            audio_data = torch_audio
            audio_queue.put_nowait(audio_data)
            i += 1
            print("recorded the " + str(i) + "th audio")


###############################################################################################################
# The transcribe_forever function receives two queues: 
# - audio_queue, which contains the audio data to be transcribed, and 
# - result_queue, which is used to store the transcribed text.
###############################################################################################################
def transcribe_forever(audio_queue, result_queue, audio_model, english, 
           wake_word, verbose):
    i = 0
    while True:
        #####################################################################
        # The function starts by getting the next audio data from the 
        # audio_queue using the audio_queue.get() method. 
        #####################################################################
        audio_data = audio_queue.get()
        
        #####################################################################
        # - If the english flag is set to True, the audio_model.transcribe()
        #   method is called with the language='english' argument to
        #   transcribe the English language audio data.
        # - If the english flag is set to False, the audio_model.transcribe()
        #   method is called without any language argument, which allows the
        #   function to automatically detect the language of the audio.
        #####################################################################
        if english:
            result = audio_model.transcribe(audio_data, language='english')
        else:
            result = audio_model.transcribe(audio_data)
        
        #####################################################################
        # - The resulting result dictionary contains several keys, one of
        #   which is "text" which simply contains the transcript of the audio.
        # - The predicted_text variable is assigned the value of the "text" key. 
        #####################################################################
        predicted_text = result["text"]
        
        #####################################################################
        # - If the predicted_text string starts with the wake_word, the
        #   function processes the text by removing the wake_word from the
        #   start of the string using regular expressions.
        #####################################################################
        if predicted_text.strip().lower().startswith(wake_word.strip().lower()):
            pattern = re.compile(re.escape(wake_word), re.IGNORECASE)
            predicted_text = pattern.sub("", predicted_text).strip()

            #################################################################
            # - Additionally, punctuation marks are removed from the
            #   predicted_text string.
            # - When you say “Hey Computer <your request>”, it may sometimes 
            #   be transcribed as “Hey Computer, <your request>”. 
            #   + Since we are going to remove the wake word, what remains 
            #     is “, <your request>”. 
            #   + Therefore, we need to remove the comma (“,”). 
            #   + Similarly, if instead of the comma we have a point, 
            #     an exclamation mark, or a question mark, we need to remove 
            #     it to only pass the request text.
            #################################################################
            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            predicted_text = predicted_text.translate({ord(i): 
                    None for i in punc})
            
            #################################################################
            # If the verbose flag is set to True, a message is printed to 
            # indicate that the wake word was detected and the text 
            # is being processed. 
            #################################################################
            if verbose:
                print("You said the wake word.. Processing {}...".
                       format(predicted_text))
            
            #################################################################
            # Finally, the predicted_text string is added to the 
            # result_queue using the result_queue.put_nowait() method.
            #################################################################
            result_queue.put_nowait(predicted_text)
        else:
            if verbose:
                print("You did not say the wake word.. Ignoring")
        print("transcribed the " + str(i) + "th audio, the audio content is : " + predicted_text)
        i += 1

############################################################################
# Reply to the user
# - The code explains the process of generating better answers using 
#   the reply() function.
# - Suggests potential future enhancements and improvements to the prototype:
#   + Creating voice embeddings for user input.
#   + Streaming audio responses instead of saving to disk.
#   + Implementing caching.
#   + Adding a stop word for immediate interruption.
############################################################################
def reply(result_queue, verbose):
    i = 0
    while True:
        question = result_queue.get()

        ####################################################################
        # We use the following format for the prompt: "Q: <question>?\nA:"
        ####################################################################
        prompt = "Q: {}?\nA:".format(question)
        
        data = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=0.5,
            max_tokens=100,
            n=1,
            stop=["\n"]
        )

        ####################################################################
        # We catch the exception in case there is no answer
        ####################################################################
        try:
            answer = data["choices"][0]["text"]
            mp3_obj = gTTS(text=answer, lang="en", slow=False)
        ####################################################################
        # In cases where the OpenAI model does not provide an answer, a set 
        # of predefined response choices is provided.
        ####################################################################
        except Exception as e:
            choices = [
                "I'm sorry, I don't know the answer to that",
                "I'm not sure I understand",
                "I'm not sure I can answer that",
                "Please repeat the question in a different way"
            ]
            mp3_obj = gTTS(text=choices[np.random.randint(0, len(choices))], 
                           lang="en", slow=False)
            if verbose:
                print(e)

        ####################################################################
        # - In both cases, we play the audio
        # - Suggests potential future enhancements and improvements 
        #   to the prototype:
        #   + Creating voice embeddings for user input.
        #   + Streaming audio responses instead of saving to disk.
        #   + Implementing caching.
        #   + Adding a stop word for immediate interruption.
        ####################################################################
        mp3_obj.save("reply.mp3")
        reply_audio = AudioSegment.from_mp3("reply.mp3")
        play(reply_audio)
        print("replied the " + str(i) + "th audio, the answer content is " + answer)
        i += 1

# ###############################################################################################################
# # The function loops continuously to wait for results from 
# # the result_queue passed as an argument.
# ###############################################################################################################
# def reply(result_queue):
#     i = 0
#     while True:
#         result = result_queue.get()

#         ############################################################
#         # Language Model Call
#         #
#         # + When a result is received, it is used as input to the 
#         #   language model (Davinci) to generate a response. 
#         # + The openai.Completion.create() method is called with the 
#         #   following arguments:
#         #   - model: the ID of the language model to use.
#         #   - prompt: the input text to use for generating the response.
#         #   - temperature: a hyperparameter that controls the degree 
#         #     of randomness in the generated response. 
#         #     + Here, it is set to 0, meaning that the response will 
#         #       be deterministic.
#         #   - max_tokens: the maximum number of tokens (words and 
#         #   punctuation) in the generated response. 
#         #     + Here, it is set to 150.
#         ############################################################
#         data = openai.Completion.create(
#             model="text-davinci-002",
#             prompt=result,
#             temperature=0,
#             max_tokens=150,
#         )

#         ############################################################
#         # Response Extraction
#         #
#         # The response generated by the language model is extracted 
#         # from the data object returned by the 
#         # openai.Completion.create() method. 
#         #  - It is stored in the answer variable.
#         ############################################################
#         answer = data["choices"][0]["text"]

#         ############################################################
#         # Text-to-Speech Conversio
#         #
#         # The gTTS (Google Text-to-Speech) module is used to convert 
#         # the generated text response into an audio file. 
#         # - The gTTS method is called with the following arguments:
#         #   + text: the generated response text.
#         #   + lang: the language of the generated response. 
#         #     - Here, it is set to English.
#         #   + slow: a boolean flag that determines whether the 
#         #     speech is slow or normal. 
#         #     - Here, it is set to False, meaning that the speech 
#         #       will be at a normal speed.
#         ############################################################
#         mp3_obj = gTTS(text=answer, lang="en", slow=False)


#         ############################################################
#         # The audio file is saved to disk as “reply.mp3”.
#         #
#         ############################################################
#         mp3_obj.save("reply.mp3")

#         ############################################################
#         # Audio Playback
#         #
#         # The “reply.mp3” file is loaded into an AudioSegment object 
#         # using the AudioSegment.from_mp3() method. 
#         # - The play() method from the pydub.playback module is used 
#         #   to play back the audio file to the user.
#         ############################################################
#         reply_audio = AudioSegment.from_mp3("reply.mp3")
#         play(reply_audio)

#         ############################################################
#         # The “reply.mp3” file is deleted using the os.remove() 
#         # method.
#         ############################################################
#         os.remove("reply.mp3")
#         print("replied the " + str(i) + "th audio, the answer content is " + answer)
#         i += 1


###############################################################################################################
# The function loops continuously to wait for results from 
# the result_queue passed as an argument.
###############################################################################################################

###############################################################################################################
# Call the init_api() function to initialize API credentials 
# using data from a ".env" file.
###############################################################################################################
init_api()

###############################################################################################################
# Call the main() function to start the application.
###############################################################################################################
main()
