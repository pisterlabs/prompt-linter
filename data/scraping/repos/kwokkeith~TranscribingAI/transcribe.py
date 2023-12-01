#! python3.7
import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

# For seamlessM4T
from seamless_communication.models.inference import Translator

# For Helsinki-NLP
from transformers import MarianMTModel, MarianTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large", help="Transcription Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Indicates if non-english model should be used")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                            "consider it a new line in the transcription.", type=float)
    parser.add_argument("--language", default='en',
                        help="Language in two character syllable to translate, defaulted to en but cn if --non_english flag is true")
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                "Run this with 'list' to view available Microphones.", type=str)
    parser.add_argument("--output", action="store_true",
                        help="To indicate if an output txt should be produced")
    parser.add_argument("--translationModel", default="helsinki",
                        help="helsinki (lightweight model) OR seamless")
    parser.add_argument("--gpu", action='store_true',
                        help="Whether gpu should be used")
    parser.add_argument("--cores", default="8",
                        help="number of cores to run models")
    args = parser.parse_args()


    # Torch configuration for number of threads
    torch.set_num_threads(int(args.cores))

    # Device
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    
    # Load Model
    audio_model = whisper.load_model(model)
    
    # Set Language Option
    if args.language != "en":
        language = args.language
    else:
        if args.non_english:
            language = "zh"
        else:
            language = args.language


    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)
    
    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user Whisper model is ready.
    print("Whisper Model loaded.")

    # *********************************************************#
    ## using Mistral-7b-openorca translator
    ## Load Mistral-7b-openorca for translation
    #template = """Translate the Chinese text that is delimited by triple backticks into English. text: ```{original_text}```"""
    #prompt_Template = PromptTemplate(template=template, input_variables=["original_text"])
    ## Callbacks support token-wise streaming
    #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    #
    #llm = LlamaCpp(
        #model_path="./Models/mistral-7b-openorca.Q5_K_M.gguf",
        #temperature=0,
        #max_tokens=2000,
        #n_ctx=8000,
        #top_p=1,
        #callback_manager=callback_manager,
        #verbose=False,  # Verbose is required to pass to the callback manager
    #)

    ### Cue the user Mistral-7b-openorca model is ready.
    ##print("Mistral-7b-openorca Model loaded.\n")

    # *********************************************************#
    if args.translationModel == "seamless":
        # Using SeamlessM4T translator
        # Initialize a Translator object with a multitask model, vocoder on the GPU.
        translator = Translator("seamlessM4T_large", vocoder_name_or_card="vocoder_36langs", device=device)
        print("seamlessM4T Model loaded.\n")
    # *********************************************************#
    # Using Helsinki-NLP model
    elif args.translationModel == "helsinki":
        model_name = "Helsinki-NLP/opus-mt-zh-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        print("Helsinki-NLP model loaded\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(temp_file, language=language, fp16=torch.cuda.is_available())

                # Translate transcription using mistral-7b-openorca model
                if args.non_english:
                    if args.translationModel == "seamles":
                        # SeamlessM4T
                        original_text = result['text']
                        text,_,_ = translator.predict(original_text.strip(), "t2tt", "eng", src_lang="cmn")
                        text = str(text)

                    ## Mistral-7b-openorca
                    #prompt = prompt_Template.format(original_text=result['text'].strip())
                    #text = llm(prompt)

                    elif args.translationModel == "helsinki":
                        # Helsinki-NLP
                        original_text = result['text'].strip()
                        translated = model.generate(**tokenizer(original_text, return_tensors="pt", padding=True))
                        text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
                        text = " ".join(text)
                else:
                    text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    if args.non_english:
                        transcription.append(f"\n-------Next Phrase-------")
                        transcription.append(f"Text: {original_text}")
                        transcription.append(f"Translated: {text}")
                    else:
                        transcription.append(text)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                # Flush stdout.
                print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            # write to file if output is true
            if args.output:
                with open("Transcription-output.txt", "w+") as f:
                    f.write("\n".join(transcription))
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
