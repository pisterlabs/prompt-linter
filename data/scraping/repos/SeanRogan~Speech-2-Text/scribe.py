import glob
import os
import datetime
import time
import openai
import constants
import logging

logging.basicConfig(level=logging.DEBUG, filename='log.txt')
start = datetime.datetime.now()     # timestamp for perf testing
last_processed_time = 0     # last processed file timestamp
transcribed = []        # list to hold files already transcribed to avoid duplicate api calls

while True:
    files = glob.glob("./recordings/*")     # files are in the recordings dir
    unprocessed_files = [f for f in files if os.path.getctime(f) > last_processed_time]    # unprocessed_files are any with a timestamp more recent than the last_processed_file's timestamp
    if not unprocessed_files:
        time.sleep(1)       # if there aren't any files to process, sleep 1second and restart loop
        continue

    latest = max(unprocessed_files, key=os.path.getctime)   # sort unprocessed files by timestamp to find latest recording

    logging.debug(str(latest))
    latest_filename = latest.split('/')[2]  # split file id from dir id
    logging.debug("latest filename is : " + str(latest_filename))

    if latest not in transcribed and os.path.exists(latest):  # if file path exists and is not present in the 'transcribed' list of files already processed..
        with open(latest, 'rb') as file:  # open file..
            logging.debug('calling api with file: ' + str(latest))
            result = openai.Audio.transcribe("whisper-1", file,
                                             api_key=constants.OPENAI_API_KEY)  # OpenAI API call, args include whisper model, sending the file object,and the api key
        logging.debug("api call success")
        print(str(result.text))  # print the results of the api call

        # todo handler service to deal with results of api call

        # record end-time time stamp, prints api call delay time for testing/eval
        end = datetime.datetime.now()
        runtime = end - start
        logging.debug('runtime for scribe loop was : ' + str(runtime))
        # append text to transcript file
        trans_ts_start = datetime.datetime.now()
        with open('transcription.txt', 'a') as f:
            f.write(result.text)
        transcribed.append(latest)
        trans_ts_end = datetime.datetime.now()
        transcribed_time = trans_ts_end - trans_ts_start
        logging.debug('transcription took :' + str(transcribed_time))
        # save list of transcribed recordings so that we don't transcribe the same one again
    last_processed_time = os.path.getctime(latest)          # record file timestamp as new last_processed_file
