import os
import sys; sys.path.append('./')
import click
import time
import openai
import queue
from threading import Thread
from fajrGPT.utils import *

openai.api_key = os.getenv("OPENAI_API_KEY")
check_ffmpeg()

@click.command()
@click.option('--countdown-time', required=True, help='Countdown time in format [number][h/m/s], i.e. 1h would create a 1 hour timer.')
@click.option('--surah', required=False, help='Specific Surah from the Quran to play for the alarm audio (int between 1 and 114). Default is the first chapter (Surah Al-Fatihah).', default=1)
@click.option('--names-flag', required=False, help='Whether or not to include a randomly selected name of Allah in the preamble.', default=True)
@click.option('--english', required=False, help='Whether or not to play audio with the english translation of the Quran verses.', default=False)
@click.option('--low-pass', required=False, help='Amount of low-pass to apply to the audio (float (KHz) or None). Default is 10 (KHz).', default=10)
@click.option('--gpt-model-type', required=False, help='Which GPT model to use for the prompt responses from OpenAI.', default="gpt-4-0314")
@click.option('--telegraphic', required=False, help='Whether or not to use a telegraphic (i.e. very simple) speech style in the response.', default=True)

def main(countdown_time, surah=1, names_flag=True, english=False, low_pass=10, gpt_model_type="gpt-4-0314", telegraphic=True):
    # initialize the result queues
    allah_queue = queue.Queue() if names_flag else None
    selected_verses_queue = queue.Queue()
    verses_explanations_queue = queue.Queue()
    quran_audio_queue = queue.Queue()
    alarm_out_queue = queue.Queue()

    # convert time to seconds
    countdown_seconds = convert_to_seconds(countdown_time)

    # Create threads for audio processing and countdown
    prepare_alarm_audio_thread = Thread(target=alarm_audio_processing, args=(surah, english, low_pass, alarm_out_queue))
    countdown_thread = Thread(target=countdown, args=(countdown_seconds,))

    # create threads for obtaining quran verse and Allah name explanations
    selected_verses_thread = Thread(target=select_quran_verse,args=(selected_verses_queue,))
    get_name_of_allah_thread = Thread(target=get_name_of_allah_and_explanation, args=(gpt_model_type,allah_queue,telegraphic)) if names_flag else None

    # Start the threads
    countdown_thread.start()
    selected_verses_thread.start()

    # wait for the selected verse thread to finish
    selected_verses_thread.join()

    # fetch the results from the queue
    verses_Quran_Module, selected_verses = selected_verses_queue.get()

    # for the selected verses, get the explanations and corresponding audio on a separate thread
    get_explanations_thread = Thread(target=get_explanations, args=(verses_Quran_Module,selected_verses,countdown_seconds,gpt_model_type,verses_explanations_queue,telegraphic))
    prepare_selected_verse_audio_thread = Thread(target=download_quran_verses_audio,args=(selected_verses,quran_audio_queue,))

    # wait a second before starting the explanations and audio downloading threads
    time.sleep(1)
    get_name_of_allah_thread.start() if names_flag else None
    get_explanations_thread.start()
    prepare_alarm_audio_thread.start()
    prepare_selected_verse_audio_thread.start()

    # wait for the explanations threads to finish
    get_name_of_allah_thread.join() if names_flag else None
    get_explanations_thread.join()
    prepare_alarm_audio_thread.join()

    # fetch the results from the queue
    name_of_allah_arabic, name_of_allah_transliteration, name_of_allah_english, explanation = allah_queue.get() if names_flag else None
    verse_texts, explanations, verses = verses_explanations_queue.get()
    selected_quran_audio_file = quran_audio_queue.get()[0]
    alarm_output_file = alarm_out_queue.get()[0] + '.mp3'

    # process the selected quran audio file on a separate thread
    filter_selected_audio_thread = Thread(target=apply_low_pass_filter,args=( selected_quran_audio_file, float(low_pass * 1000) ))
    filter_selected_audio_thread.start()
    filter_selected_audio_thread.join()

    # Wait for both threads to finish
    countdown_thread.join()

    # Play alarm audio with fade-in effect on a separate thread
    play_audio_thread = Thread(target=play_audio, args=(alarm_output_file,))
    play_audio_thread.start()

    # display the name of Allah and explanation
    display_allah_name_and_explanation(name_of_allah_arabic, name_of_allah_transliteration, name_of_allah_english, explanation) if names_flag else None

    print(f'When you are ready to see the selected Quran verses, press ENTER.')
    input()

    # print the selected verses
    print_selected_verses(verses)

    # stop the misharay audio once the user has finished reading the name of Allah
    stop_audio(5,)

    # play the audio of the quran verses with 5 second fade in
    play_audio_thread = Thread(target=play_audio, args=(selected_quran_audio_file,))
    play_audio_thread.start()

    # display the explanations
    display_quran_verse_explanations(verse_texts,explanations,verses)

    # stop the audio once the user has completed reading the verses
    stop_audio(5,)

    # return back to the main thread
    play_audio_thread.join()

if __name__ == "__main__":
    main()

