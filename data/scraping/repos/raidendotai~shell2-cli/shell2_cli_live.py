import os, json, sys, string, time, time
import asyncio, threading, argparse
from slugify import slugify
import configparser

from rich.console import Console
from prompt_toolkit.styles import Style
from prompt_toolkit import prompt
from prompt_toolkit.shortcuts import clear
from prettyprinter import cpprint
import colorama

import requests
from retry import retry
from requests.exceptions import RequestException

from shell2.client import Shell2Client
from google.cloud import firestore
from google.oauth2.credentials import Credentials


config = configparser.ConfigParser()
CONFIG_DIR = os.path.expanduser("~/")
CONFIG_FILE = os.path.join(CONFIG_DIR, ".shell2_cli_config")

config.read(CONFIG_FILE)
def get_api_key():
    return config['DEFAULT'].get('apikey', None)




################################################
# VOCAL COMMAND STUFF
# https://stackoverflow.com/questions/46734345/python-record-on-loop-stop-recording-when-silent

from rhasspysilence import WebRtcVadRecorder,VoiceCommand, VoiceCommandResult
import dataclasses
import typing
from queue import Queue
import io
from pathlib import Path
import shlex
import wave
import subprocess
import pyaudio

import openai

pa =pyaudio.PyAudio()
#you can change the options (these are default settings)
vad_mode = 3
sample_rate = 16000
min_seconds= 0.3
max_seconds = 60
speech_seconds = 0.5 #0.3
silence_seconds = 3 #0.5
before_seconds = 0.5
chunk_size= 960
skip_seconds = 0
audio_source = None
channels =1
def SpeechToText():
    recorder = WebRtcVadRecorder(
        vad_mode=vad_mode,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        speech_seconds=speech_seconds,
        silence_seconds=silence_seconds
    )
    recorder.start()
    # file directory
    wav_sink = './'
    wav_dir = None
    # file name
    wav_filename = '_temp_vocal_recording'
    if wav_sink:
        wav_sink_path = Path(wav_sink)
        if wav_sink_path.is_dir():
            # Directory to write WAV files
            wav_dir = wav_sink_path
        else:
            # Single WAV file to write
            wav_sink = open(wav_sink, "wb")
    voice_command: typing.Optional[VoiceCommand] = None
    audio_source = pa.open(rate=sample_rate,format=pyaudio.paInt16,channels=channels,input=True,frames_per_buffer=chunk_size)
    audio_source.start_stream()
    # print("Ready", file=sys.stderr)
    def buffer_to_wav(buffer: bytes) -> bytes:
        """Wraps a buffer of raw audio data in a WAV"""
        rate = int(sample_rate)
        width = int(2)
        channels = int(1)
 
        with io.BytesIO() as wav_buffer:
            wav_file: wave.Wave_write = wave.open(wav_buffer, mode="wb")
            with wav_file:
                wav_file.setframerate(rate)
                wav_file.setsampwidth(width)
                wav_file.setnchannels(channels)
                wav_file.writeframesraw(buffer)
 
            return wav_buffer.getvalue()
    try:
        chunk = audio_source.read(chunk_size)
        while chunk:
 
            # Look for speech/silence
            voice_command = recorder.process_chunk(chunk)
 
            if voice_command:
                is_timeout = voice_command.result == VoiceCommandResult.FAILURE
                # Reset
                audio_data = recorder.stop()
                if wav_dir:
                    # Write WAV to directory
                    wav_path = (wav_dir / time.strftime(wav_filename)).with_suffix(
                        ".wav"
                    )
                    wav_bytes = buffer_to_wav(audio_data)
                    wav_path.write_bytes(wav_bytes)
                    #print(wav_path)
                    #print('file saved')
                    break
                elif wav_sink:
                    # Write to WAV file
                    wav_bytes = core.buffer_to_wav(audio_data)
                    wav_sink.write(wav_bytes)
            # Next audio chunk
            chunk = audio_source.read(chunk_size)
 
    finally:
        try:
            audio_source.close_stream()
        except Exception:
            pass




################################################






firestoreClient = None

loop = asyncio.get_event_loop()
console = Console()


colorama.init(autoreset=True)
def _streamprint(message):
    print(message,end='\r')

API_KEY = get_api_key()
SHELL2_CLIENT = Shell2Client(API_KEY)
SESSION_ID = False
SESSION_USER = False
SANDBOX_TYPE = False
FOLDER_STATE_7Z = False
FOLDER_STATE_NOSYNC = False
USER_INPUT_MODE = 'text'

STACK_TRACK = {}
FILES_TRACK = {}



def _vocal_input():
    SpeechToText()
    
    loop.call_soon_threadsafe(console.print, (f"[red]transcribing audio[/red]")  )
    whisper_response = openai.Audio.transcribe("whisper-1", open( '_temp_vocal_recording.wav' , "rb") )
    transcript = whisper_response['text']
    os.remove( '_temp_vocal_recording.wav' )
    transcript_alpha_only = ''.join(filter(str.isalpha, transcript.lower().strip() ))
    if transcript_alpha_only == 'thankyou':
        return '/done'
    if transcript.lower().strip().startswith('yo') :
        return f'/m { " ".join( transcript.split(" ")[1:] ) }'        
    return transcript


def _save_json(path,obj):
    with open(path,'w') as fout:
        json.dump(obj, fout)

@retry(RequestException, tries=5, delay=1, backoff=2)
def _postWithRetry(url,body,headers):
    response = requests.post(
        url,
        json=body,
        headers=headers,
    )
    response.raise_for_status()
    return json.loads(response.text)

def _list_files_current_dir():
    file_list = []
    total_size = 0
    
    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
            total_size += os.path.getsize(file_path)
    
    if SANDBOX_TYPE == 'sequence':
        file_list = [f for f in file_list if 'sequence.txt' not in f]
    
    if total_size > 500 * 1024 * 1024:  # 500MB in bytes
        print(f'> total file sizes of folder over 500Mb, will not upload current folder files')
        return False
    
    if not file_list:
        return False
    
    return file_list

def make_7z_current_dir():
    import py7zr
    print('> found files in current folder; preparing archive to sync files to shell2 sandbox')
    current_folder = os.getcwd()
    archive_name = slugify( os.path.basename(current_folder) ) + ".7z"
    
    with py7zr.SevenZipFile(archive_name, 'w') as archive:
        for root, _, files in os.walk(current_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if SANDBOX_TYPE == 'sequence':
                    if ( file_path != os.path.abspath(archive_name) ) and ( 'sequence.txt' not in file_path ):  # Skip adding the archive file itself
                        arcname = os.path.relpath(file_path, current_folder)
                        archive.write(file_path, arcname=arcname)
                else:
                    if ( file_path != os.path.abspath(archive_name) ) :  # Skip adding the archive file itself
                        arcname = os.path.relpath(file_path, current_folder)
                        archive.write(file_path, arcname=arcname)                
    return archive_name

def sync_current_dir():
    global SESSION_ID
    global SESSION_USER
    global SANDBOX_TYPE
    global SHELL2_CLIENT
    global FOLDER_STATE_7Z
    global FILES_TRACK
    try:
        response = SHELL2_CLIENT.storage.upload({
            'filepath': './' + FOLDER_STATE_7Z,
        })
    except Exception as e:
        msg_obj = {
            'upload_limits' : 'unable to upload. if you are a trial user, uploads storage is limited.'
        }
        console.print( f'[red]{json.dumps(  msg_obj , indent=1 )}[/red]' )
        os.remove(FOLDER_STATE_7Z)
        return
    if not response['status']:
        msg_obj = {
            'upload_limits' : 'unable to upload. if you are a trial user, uploads storage is limited.'
        }    
        console.print( f'[red]{json.dumps(  msg_obj , indent=1 )}[/red]' )
        os.remove(FOLDER_STATE_7Z)
        return
    
    os.remove(FOLDER_STATE_7Z)
    
    
    # mechanism to not redownload
    import hashlib
    for f in _list_files_current_dir():
        md5 = hashlib.md5(open(f,'rb').read()).hexdigest()
        FILES_TRACK[md5] = f
    
    # msg call /file_extract
    if SANDBOX_TYPE == 'session':
        SHELL2_CLIENT.session.message({
            'sessionId' : SESSION_ID,
            'message': f'/file_extract uploads://{FOLDER_STATE_7Z} {FOLDER_STATE_7Z}',
            'user':SESSION_USER
        })

    

DOWNLOADING_FILES = {}
def download_file(out_path,url):
    global DOWNLOADING_FILES
    DOWNLOADING_FILES[out_path] = False
    if len(out_path.split('/')) > 1:
        os.makedirs(
            os.path.join(
                os.getcwd(),
                '/'.join( out_path.split('/')[:-1] )
            ),
            exist_ok=True
        )
    r = requests.get( url , allow_redirects=True)
    open( os.path.join(os.getcwd(), out_path) , 'wb').write(r.content)
    DOWNLOADING_FILES[out_path] = True
    loop.call_soon_threadsafe(console.print, (f"[red]> completed :[/red] {out_path}")  )

async def stack_listener(firestorePath):
    global SANDBOX_TYPE
    global STACK_TRACK
    global FILES_TRACK
    global SHELL2_CLIENT
    collection_ref = firestoreClient.collection(firestorePath)
    collection_ref = collection_ref.order_by("timestampCreated", direction=firestore.Query.DESCENDING).limit(1)

    def on_collection_snapshot(col_snapshot, changes, read_time):
        for doc_snapshot in col_snapshot:
            doc_data = doc_snapshot.to_dict()
            if doc_data:
                stack_entry = doc_data
                if not ( stack_entry['type'].startswith('stream_') ):
                    if '_extract' in stack_entry['type']:
                        stack_entry['data']['text'] = stack_entry['data']['text'][:500] + '...'
                        try:
                            loop.call_soon_threadsafe(console.print, (f"[red]stack:[/red] {json.dumps( stack_entry, indent=1)}")  )
                        except Exception as e:
                            False
                    elif stack_entry['type'] == 'prompt_code':
                        del stack_entry['data']['prompt']
                        try:
                            loop.call_soon_threadsafe(console.print, f"[red]stack:[/red] {json.dumps( stack_entry, indent=1)}" )
                        except Exception as e:
                            False
                    elif stack_entry['type'] == 'generated_text':
                        loop.call_soon_threadsafe(
                            console.print,
                            f"\n[blue]________________________________[/blue]\n[red]answer :[/red]\n```\n{stack_entry['data']}\n```\n[blue]________________________________[/blue]"
                        )
                    elif stack_entry['type'] == 'generated_code':
                        loop.call_soon_threadsafe(
                            console.print,
                            f"\n[blue]________________________________[/blue]\n[red]code :[/red]\n```\n{stack_entry['data']['code']}\n```\n\n[red]comments :\n```\n{stack_entry['data']['comments']}\n```[/red]\n[blue]________________________________[/blue]"
                        )
                    elif stack_entry['type'] == 'execution_shell':
                        loop.call_soon_threadsafe(
                            console.print,
                            f"\n[blue]________________________________[/blue]\n[red]command :[/red]\n```\n{stack_entry['data']['command']}\n```\n\n[red]execution :\n\n{stack_entry['data']['output']}\n[/red]\n[blue]________________________________[/blue]"
                        )
                    elif stack_entry['type'] == 'execution_success':
                        loop.call_soon_threadsafe(
                            console.print,
                            f"\n[blue]________________________________[/blue]\n[red]execution success :[/red]\n{stack_entry['data']}\n[blue]________________________________[/blue]"
                        )
                    elif stack_entry['type'] == 'execution_fail':
                        loop.call_soon_threadsafe(
                            console.print,
                            f"\n[blue]________________________________[/blue]\n[red]execution fail :[/red]\n{stack_entry['data']}\n[blue]________________________________[/blue]"
                        )                               
                    else:
                        try:
                            loop.call_soon_threadsafe(console.print, f"[red]stack:[/red] {json.dumps( stack_entry, indent=1)}" )
                        except Exception as e:
                            False
                        if stack_entry['type'] == 'files_state':
                            
                            
                            json_body = {'sessionId' : SESSION_ID}
                            if SANDBOX_TYPE == 'sequence':
                                json_body = {'sequenceId' : SESSION_ID} # we gon need to fix this later
                            
                            if SESSION_USER : json_body['user'] = SESSION_USER
                            filesState = SHELL2_CLIENT.storage.download(json_body)

                            for newfile in filesState['files']:
                                #print(newfile)
                                if newfile['md5'] not in FILES_TRACK:
                                    FILES_TRACK[ newfile['md5'] ] = newfile['file']
                                    # out_path = f'shell2_data/session/{SESSION_ID}/' + newfile['file']
                                    out_path = newfile['file']
                                    loop.call_soon_threadsafe(console.print, (f"[red]> downloading :[/red] {out_path}")  )
                                    
                                    download_thread = threading.Thread(target=download_file, args=(out_path, newfile['url'] ))
                                    download_thread.start()
                                    # download_thread.join()
                                                    
                
                else:
                    try:
                        if stack_entry['type'] == 'stream_session_resume_files':
                            if doc_snapshot.id not in STACK_TRACK:
                                loop.call_soon_threadsafe(console.print, f"[red]the sandbox is resuming files from previous data ...[/red]" )
                                STACK_TRACK[doc_snapshot.id] = stack_entry['data']
                        elif stack_entry['type'] == 'stream_download':
                            if doc_snapshot.id not in STACK_TRACK:
                                loop.call_soon_threadsafe(console.print, f"[red]the sandbox is downloading new files ...[/red]" )
                                STACK_TRACK[doc_snapshot.id] = stack_entry['data']
                        else:

                            if doc_snapshot.id not in STACK_TRACK:
                                # loop.call_soon_threadsafe(console.print, f"[red]processing ...[/red]" )
                                STACK_TRACK[doc_snapshot.id] = stack_entry

                            
                            stream_progress = ''
                            stream_previous = ''
                            if stack_entry['type'] == 'stream_code':
                                stream_progress = '> code :\n````\n' + stack_entry['data']['code']
                                if len( stack_entry['data']['comments'] ):
                                    stream_progress += '\n```\n' + stack_entry['data']['comments']
                                
                                stream_previous = '> code :\n````\n' + STACK_TRACK[doc_snapshot.id]['data']['code']
                                if len( STACK_TRACK[doc_snapshot.id]['data']['comments'] ):
                                    stream_previous += '\n```\n' + STACK_TRACK[doc_snapshot.id]['data']['comments']
                                    
                            elif stack_entry['type'] == 'stream_text':
                                stream_progress = '> assistant : ' + stack_entry['data']
                                stream_previous = '> assistant : ' + STACK_TRACK[doc_snapshot.id]['data']
                            elif stack_entry['type'] == 'stream_shell':
                                stream_progress = '> execution : ' + stack_entry['data']
                                stream_previous = '> execution : ' + STACK_TRACK[doc_snapshot.id]['data']
                            
                            loop.call_soon_threadsafe(console.print, f"[blue]{stream_progress[ len(stream_previous) : ] }[/blue]" )
                            
                            STACK_TRACK[doc_snapshot.id] = stack_entry
                    except Exception as e:
                        False

    col_listener = collection_ref.on_snapshot(on_collection_snapshot)

MESSAGE_TRACK = {}
SESSION_START_TIME = 0
async def message_listener(firestorePath):
    global MESSAGE_TRACK
    collection_ref = firestoreClient.collection(firestorePath)
    collection_ref = collection_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1)

    def on_collection_snapshot(col_snapshot, changes, read_time):
        for doc_snapshot in col_snapshot:
            doc_data = doc_snapshot.to_dict()
            if doc_data['timestamp'] > SESSION_START_TIME:
                if doc_snapshot.id not in MESSAGE_TRACK:
                    MESSAGE_TRACK[doc_snapshot.id] = True
                    message = doc_data
                    loop.call_soon_threadsafe(console.print, f"[blue]message:[/blue] {json.dumps( message, indent=1)}" )

    col_listener = collection_ref.on_snapshot(on_collection_snapshot)


SESSION_BUSY = True
SESSION_DONE = False
async def session_metadata_listener(firestorePath):
    global SESSION_BUSY
    global SESSION_DONE
    global DOWNLOADING_FILES
    global SHELL2_CLIENT
    def session_metadata_onsnapshot(doc_snapshot, changes, read_time):
        global SESSION_BUSY
        global SESSION_DONE
        global DOWNLOADING_FILES
        global SHELL2_CLIENT
        updatedData = None
        for change in changes:
            if change.type.name == 'ADDED' or change.type.name == 'MODIFIED':
                updatedData = change.document.to_dict()
                if 'busy' in updatedData: SESSION_BUSY = updatedData['busy']
                if 'done' in updatedData:
                    SESSION_DONE = updatedData['done']
                    if SESSION_DONE:
                        try:
                            print('> sandbox done, waiting to check for pending file downloads & closing')
                            time.sleep(2)
                            downloads_done_check = [e for e in list(DOWNLOADING_FILES.values()) if not e]
                            while len(downloads_done_check):
                                time.sleep(1)
                                downloads_done_check = [e for e in list(DOWNLOADING_FILES.values()) if not e]
                            time.sleep(0.5)
                            loop.stop()  # Stop the event loop
                            loop.close()  # Close the event loop
                        except Exception as e:
                            False
                            
                        response_sequence = False
                        if SANDBOX_TYPE == 'sequence':
                            response_sequence = SHELL2_CLIENT.sequence.get({"sequenceId" : SESSION_ID})
                            _save_json(
                                os.path.join(os.getcwd(), f'sequenceId_{SESSION_ID}.json'),
                                response_sequence
                            )
                            cpprint({
                                'sequenceId' : SESSION_ID,
                                'saved' : f"./sequenceId_{SESSION_ID}.json"
                            })
                        elif SANDBOX_TYPE == 'session':
                            response_sequence = SHELL2_CLIENT.session.get({"sessionId" : SESSION_ID})
                            _save_json(
                                os.path.join(os.getcwd(), f'sessionId_{SESSION_ID}.json'),
                                response_sequence
                            )
                            cpprint({
                                'sessionId' : SESSION_ID,
                                'saved' : f"./sessionId_{SESSION_ID}.json"
                            })
                            
                        os._exit(0)
    firestoreClient.document(firestorePath).on_snapshot(session_metadata_onsnapshot)

def user_input():
    global SESSION_BUSY
    global SESSION_DONE
    global SESSION_ID
    global SESSION_USER
    global SHELL2_CLIENT
    global API_KEY
    global USER_INPUT_MODE
    
    while True:
        if (not SESSION_BUSY) and (not SESSION_DONE):
            time.sleep(0.75)
            
            if USER_INPUT_MODE == 'text':
                user_message = prompt("user > ", style=Style.from_dict({"prompt": "yellow"}))
            elif USER_INPUT_MODE == 'voice':
                loop.call_soon_threadsafe(
                    console.print,
                    ( f"[yellow]\nlistening for vocal input ____________________________\n\t- for text prompts (no code gen), start by saying `yo`\n\t- if you're done, say `thank you` to close session\nspeak >[/yellow]" )
                )
                user_message = _vocal_input()

            if len(user_message):
                SHELL2_CLIENT.session.message({
                    'sessionId' : SESSION_ID,
                    'message':user_message,
                    'user':SESSION_USER
                })
                SESSION_BUSY = True

def session_new(timeout,multiplayer):
    global firestoreClient
    global SESSION_START_TIME
    global MESSAGE_TRACK
    global STACK_TRACK
    global SESSION_DONE
    global SESSION_ID
    global SESSION_USER
    global SANDBOX_TYPE
    global API_KEY
    global SHELL2_CLIENT
    global FOLDER_STATE_7Z
    global FOLDER_STATE_NOSYNC
    
    SESSION_START_TIME = int( time.time()*1000 )
    MESSAGE_TRACK = {}
    STACK_TRACK = {}
    SESSION_DONE = False
    SESSION_USER = False
    SANDBOX_TYPE = 'session'
    
    if not FOLDER_STATE_NOSYNC:
        if _list_files_current_dir():
            FOLDER_STATE_7Z = make_7z_current_dir()
    
    print('> creating shell2 session')
    sessionApiResponse = SHELL2_CLIENT.session.new({
        'timeout':timeout,
        'multiplayer':multiplayer
    })
    
    console.print( f'[red]{json.dumps(  sessionApiResponse , indent=1 )}[/red]' )
    if ( 'status' in sessionApiResponse ) and ( sessionApiResponse['status'] == False ):
        os._exit(0)
    
    SESSION_ID = sessionApiResponse['sessionId']
    print("### created sessionId : " + SESSION_ID)
    
    sdkStreamData = _postWithRetry(
        'https://api.shell2.raiden.ai/user/sdk/stream' ,
        {'sessionId' : SESSION_ID},
        {'key':API_KEY}
    )
    
    firestoreClient = firestore.Client(
        project = sdkStreamData['config']['projectId'],
        credentials = Credentials(sdkStreamData['token']['idToken'], sdkStreamData['token']['refreshToken']),
        client_info = sdkStreamData['config']
    )
    
    if multiplayer:
        user_email = sdkStreamData['stack'].split('/')[1]
        msg_obj = {
            'multiplayer' : 'you have enabled multiplayer. you can share this link with other shell2 users.',
            'share' : f'https://shell2.raiden.ai/view/session/{user_email}/{SESSION_ID}'
        }
        console.print( f'[red]{json.dumps(  msg_obj , indent=1 )}[/red]' )    
    
    # Start asyncio event loop
    sessionStackFirestorePath = sdkStreamData['stack']
    asyncio.ensure_future(stack_listener(sessionStackFirestorePath), loop=loop)
    
    sessionMessagesFirestorePath = sdkStreamData['messages']
    asyncio.ensure_future(message_listener(sessionMessagesFirestorePath), loop=loop)    
    #asyncio.ensure_future(busy_listener(), loop=loop)


    sessionMetadataFirestorePath = sdkStreamData['metadata']
    asyncio.ensure_future(session_metadata_listener(sessionMetadataFirestorePath), loop=loop)

    # Start user input thread
    input_thread = threading.Thread(target=user_input, daemon=True)
    input_thread.start()
    
    if not FOLDER_STATE_NOSYNC:
        if FOLDER_STATE_7Z:
            sync_current_dir()

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


def session_resume(sessionId,timeout,multiplayer):
    global firestoreClient
    global SESSION_START_TIME
    global MESSAGE_TRACK
    global STACK_TRACK
    global SESSION_DONE
    global SESSION_ID
    global SESSION_USER
    global API_KEY
    global SANDBOX_TYPE
    global SHELL2_CLIENT
    global FOLDER_STATE_7Z
    global FOLDER_STATE_NOSYNC
    
    SESSION_START_TIME = int( time.time()*1000 )
    MESSAGE_TRACK = {}
    STACK_TRACK = {}
    SESSION_DONE = False
    SANDBOX_TYPE = 'session'
    SESSION_USER = False
    
    if not FOLDER_STATE_NOSYNC:
        if _list_files_current_dir():
            FOLDER_STATE_7Z = make_7z_current_dir()    
    
    
    print('> resuming shell2 session')
    sessionApiResponse = SHELL2_CLIENT.session.resume({
        'sessionId':sessionId,
        'timeout':timeout,
        'multiplayer':multiplayer
    })

    
    console.print( f'[red]{json.dumps(  sessionApiResponse , indent=1 )}[/red]' )
    if ( 'status' in sessionApiResponse ) and ( sessionApiResponse['status'] == False ):
        os._exit(0)
    
    SESSION_ID = sessionApiResponse['sessionId']
    print("### resumed sessionId : " + SESSION_ID)
    
    
    sdkStreamData = _postWithRetry(
        'https://api.shell2.raiden.ai/user/sdk/stream' ,
        {'sessionId' : SESSION_ID},
        {'key':API_KEY}
    )
    
    firestoreClient = firestore.Client(
        project = sdkStreamData['config']['projectId'],
        credentials = Credentials(sdkStreamData['token']['idToken'], sdkStreamData['token']['refreshToken']),
        client_info = sdkStreamData['config']
    )
    
    # Start asyncio event loop
    sessionStackFirestorePath = sdkStreamData['stack']
    asyncio.ensure_future(stack_listener(sessionStackFirestorePath), loop=loop)
    
    sessionMessagesFirestorePath = sdkStreamData['messages']
    asyncio.ensure_future(message_listener(sessionMessagesFirestorePath), loop=loop)    
    #asyncio.ensure_future(busy_listener(), loop=loop)


    sessionMetadataFirestorePath = sdkStreamData['metadata']
    asyncio.ensure_future(session_metadata_listener(sessionMetadataFirestorePath), loop=loop)

    # Start user input thread
    input_thread = threading.Thread(target=user_input, daemon=True)
    input_thread.start()

    if not FOLDER_STATE_NOSYNC:
        if FOLDER_STATE_7Z:
            sync_current_dir()


    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


def session_join(sessionId,user):
    global firestoreClient
    global SESSION_START_TIME
    global MESSAGE_TRACK
    global STACK_TRACK
    global SESSION_DONE
    global SESSION_ID
    global SESSION_USER
    global API_KEY
    global SANDBOX_TYPE
    global SHELL2_CLIENT
    global FOLDER_STATE_7Z
    global FOLDER_STATE_NOSYNC
    
    SESSION_START_TIME = int( time.time()*1000 )
    MESSAGE_TRACK = {}
    STACK_TRACK = {}
    SESSION_DONE = False
    SANDBOX_TYPE = 'session'
    
    SESSION_USER = user
    SESSION_ID = sessionId
    
    if SESSION_USER:
        print("### joining sessionId : " + SESSION_ID + ' from ' + SESSION_USER)
    else:
        print("### joining sessionId : " + SESSION_ID )
    
    msg_warn = {
        'multiplayer' : 'make sure the session is still active'
    }
    console.print( f'[red]{json.dumps(  msg_warn , indent=1 )}[/red]' )
    
    if not FOLDER_STATE_NOSYNC:
        if _list_files_current_dir():
            FOLDER_STATE_7Z = make_7z_current_dir()    
    
    sdkStreamData = _postWithRetry(
        'https://api.shell2.raiden.ai/user/sdk/stream' ,
        {'sessionId' : SESSION_ID , 'user' : SESSION_USER },
        {'key':API_KEY}
    )
    
    
    if not SESSION_USER:
        SESSION_USER = sdkStreamData['stack'].split('/')[1]
    
    sdkStreamData['metadata'] = f'userdata/{SESSION_USER}/automation_layer/shell2/session/{SESSION_ID}'
    sdkStreamData['stack'] = f'userdata/{SESSION_USER}/automation_layer/shell2/session/{SESSION_ID}/stack'
    sdkStreamData['messages'] = f'userdata/{SESSION_USER}/automation_layer/shell2/session/{SESSION_ID}/message'
    
    
    # print(sdkStreamData)
    
    firestoreClient = firestore.Client(
        project = sdkStreamData['config']['projectId'],
        credentials = Credentials(sdkStreamData['token']['idToken'], sdkStreamData['token']['refreshToken']),
        client_info = sdkStreamData['config']
    )
    
    # Start asyncio event loop
    sessionStackFirestorePath = sdkStreamData['stack']
    asyncio.ensure_future(stack_listener(sessionStackFirestorePath), loop=loop)
    
    sessionMessagesFirestorePath = sdkStreamData['messages']
    asyncio.ensure_future(message_listener(sessionMessagesFirestorePath), loop=loop)    
    #asyncio.ensure_future(busy_listener(), loop=loop)


    sessionMetadataFirestorePath = sdkStreamData['metadata']
    asyncio.ensure_future(session_metadata_listener(sessionMetadataFirestorePath), loop=loop)

    # Start user input thread
    input_thread = threading.Thread(target=user_input, daemon=True)
    input_thread.start()

    if not FOLDER_STATE_NOSYNC:
        if FOLDER_STATE_7Z:
            sync_current_dir()

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


def sequence_run(timeout,sequence,webhook):
    global firestoreClient
    global SESSION_START_TIME
    global MESSAGE_TRACK
    global STACK_TRACK
    global SESSION_DONE
    global SESSION_ID
    global SESSION_USER
    global API_KEY
    global SANDBOX_TYPE
    global SHELL2_CLIENT
    global FOLDER_STATE_7Z
    global FOLDER_STATE_NOSYNC
    
    SESSION_START_TIME = int( time.time()*1000 )
    MESSAGE_TRACK = {}
    STACK_TRACK = {}
    SESSION_DONE = False
    SANDBOX_TYPE = 'sequence'
    
    if not FOLDER_STATE_NOSYNC:
        if _list_files_current_dir():
            FOLDER_STATE_7Z = make_7z_current_dir()
            sync_current_dir()
            sequence = [ f'/file_extract uploads://{FOLDER_STATE_7Z} {FOLDER_STATE_7Z}' ] + sequence
    
    print('> starting shell2 sequence')
    sequenceApiResponse = SHELL2_CLIENT.sequence.run({
        'timeout':timeout,
        'sequence':sequence,
        'webhook':webhook
    })


    console.print( f'[red]{json.dumps(  sequenceApiResponse , indent=1 )}[/red]' )
    if ( 'status' in sequenceApiResponse ) and ( sequenceApiResponse['status'] == False ):
        os._exit(0)


    SESSION_ID = sequenceApiResponse['sequenceId']
    print("### created sequenceId : " + SESSION_ID)
    
    sdkStreamData = _postWithRetry(
        'https://api.shell2.raiden.ai/user/sdk/stream' ,
        {'sequenceId' : SESSION_ID},
        {'key':API_KEY}
    )
        
    firestoreClient = firestore.Client(
        project = sdkStreamData['config']['projectId'],
        credentials = Credentials(sdkStreamData['token']['idToken'], sdkStreamData['token']['refreshToken']),
        client_info = sdkStreamData['config']
    )
    
    # Start asyncio event loop
    sessionStackFirestorePath = sdkStreamData['stack']
    asyncio.ensure_future(stack_listener(sessionStackFirestorePath), loop=loop)
    

    sessionMetadataFirestorePath = sdkStreamData['metadata']
    asyncio.ensure_future(session_metadata_listener(sessionMetadataFirestorePath), loop=loop)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()

def main():
    global FOLDER_STATE_NOSYNC
    global USER_INPUT_MODE
    
    # is called from main cli, ~like this
    
    # shell2_cli_live --sandbox session --action new --timeout 600 --multiplayer
    # shell2_cli_live --sandbox session --action resume --sessionId "someId" --timeout 600 --multiplayer
    # shell2_cli_live --sandbox session --action join --link "some_link"
    # shell2_cli_live --sandbox session --action join --sessionId example --user some@example.com
    
    # shell2_cli_live --sandbox sequence --action run --timeout 600 --webhook

    
    parser = argparse.ArgumentParser(description='shell2 live cli')
    
    parser.add_argument('--sandbox', choices=['session', 'sequence'])
    parser.add_argument('--action', choices=['new', 'resume','join','run'] )
    parser.add_argument('--timeout', type=int, required=False)
    
    # session stuff
    parser.add_argument('--sessionId', type=str, required=False)
    parser.add_argument('--multiplayer', action='store_true', required=False)
    parser.add_argument('--link', type=str, required=False) #for multiplayer
    parser.add_argument('--user', type=str, required=False)
    # sequence stuff
    parser.add_argument('--sequenceId', type=str, required=False)
    parser.add_argument('--webhook', type=str, required=False)
    
    parser.add_argument('--nosync', action='store_true', required=False)
    
    parser.add_argument('--voice', action='store_true', required=False)
    
    args = parser.parse_args()
    
    FOLDER_STATE_NOSYNC = args.nosync
    
    current_settings = SHELL2_CLIENT.settings.get()
    current_keystore = current_settings['settings']['keystore']
    current_llm = current_settings['settings']['llm']
    if current_llm.startswith('openai/') and current_keystore['openai'].startswith('sk-OPEN'):
        console_msg = {
            'llm' : current_llm,
            'missing_api_key' : 'openai',
            'error' : f"your selected LLM is `{current_llm}`, but you haven't setup an OpenAI key. set it up in your shell2 settings.",
        }
        console.print( f'[red]{json.dumps(  console_msg , indent=1 )}[/red]' )
        exit(0)
    if current_llm.startswith('replicate/') and current_keystore['replicate'].startswith('r8_REPL'):
        console_msg = {
            'llm' : current_llm,
            'missing_api_key' : 'replicate',
            'error' : f"your selected LLM is `{current_llm}`, but you haven't setup a Replicate key. set it up in your shell2 settings.",
        }
        console.print( f'[red]{json.dumps(  console_msg , indent=1 )}[/red]' )
        exit(0)
        
        
    if args.voice :
        
        console_msg = {
            'input_mode' : 'voice',
            'message' : 'voice input requires whisper by OpenAI. checking your keystore.'
        }
        
        console.print( f'[yellow]{json.dumps(  console_msg , indent=1 )}[/yellow]' )        
        
        
        

        if current_keystore['openai'].startswith( 'sk-OPEN' ) :
            console_msg = {
                'error' : 'no API key found for OpenAI. go to your shell2 settings to set it',
                'input_mode' : 'input mode now set to text'
            }
            console.print( f'[red]{json.dumps(  console_msg , indent=1 )}[/red]' )
            USER_INPUT_MODE = 'text'
        else:
            USER_INPUT_MODE = 'voice'
            openai.api_key = current_keystore['openai']
            console_msg = {
                'input_mode' : 'voice input enabled'
            }
            console.print( f'[red]{json.dumps(  console_msg , indent=1 )}[/red]' )
    
    if args.sandbox == 'session':
        if args.action == 'new':
            query = {
                'timeout' : args.timeout,
                'multiplayer': args.multiplayer,
            }
            print('## SESSION : new session ##')
            session_new(args.timeout,args.multiplayer)
        elif args.action == 'resume':
            query = {
                'sessionId' : args.sessionId,
                'timeout' : args.timeout,
                'multiplayer': args.multiplayer,
            }
            print('## SESSION : resume session ##')
            session_resume(args.sessionId,args.timeout,args.multiplayer)
        elif args.action == 'join':
            if args.link and len(args.link):
                query = {
                    'link' : args.link,
                }
                sessionId = args.link.split('/')[-1]
                user = args.link.split('/')[-2]
                session_join(sessionId,user)
                print('## SESSION : join session ##')
            elif args.sessionId and len(args.sessionId):
                if args.user and len(args.user):
                    session_join(args.sessionId,args.user)
                else:
                    session_join(args.sessionId,False)
    elif args.sandbox == 'sequence':
        if args.action == 'run':
            sequence = []
            try:
                with open( os.path.join(os.getcwd(), 'sequence.txt') , 'r') as fin:
                    sequence = fin.read().split('\n\n')
                sequence = [e for e in sequence if (e and len(e)) ]
            except Exception as e:
                print('> you need to have a sequence.txt file in your current folder')
                exit(0)
            if not len(sequence):
                print('> empty sequence, cannot run')
                exit(0)
            query = {
                'webhook' : args.webhook,
                'timeout' : args.timeout,
                'sequence' : sequence,
            }
            print('## SEQUENCE : run sequence ##')
            print({'sequence':sequence})
            sequence_run(args.timeout,sequence,args.webhook)

if __name__ == "__main__":
    main()