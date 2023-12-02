import os
import openai
import json
import shutil
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
sony_directory = os.getenv("DIR_LOCATION")
audiobackup_directory = os.getenv("COPY_AUDIO_TO_LOCATION")
textcreation_directory = os.getenv("COPY_TEXT_TO_LOCATION")
AudioSegment.converter = os.getenv("FFMPEG_LOCATION")
temp_directory = os.getenv("TEMP_LOCATION")
summary_time = (int(os.getenv("SUMMARY_TIME")) * 1000) # Seconds
segment_time = (int(os.getenv("SEGMENT_TIME")) * 1000) # Seconds

def check_sony_file_directory():
    if os.path.exists(sony_directory):
        return(True)
    else:
        return(False)

def get_files_from_sony():
    return(os.listdir(sony_directory))

def create_temp_directory():
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)

def summary_text_for_filename(summarytext):
    command = """
    You will be given notes that need to be summarized into 200 characters or less. 
    Words need an underscore between them. Do not include any date information in the summary. 
    This will need to make sense to a reader with the content produced. Here are the notes: 
    """
    
    contenttosend = command + summarytext
    
    system_content = """
    You are asummarizing notes to provide meaniingful content for naming a file.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",   
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": contenttosend}, 
         ]
    )
    return(str(response["choices"][0]["message"]["content"]))

def break_audio_into_segments(audio, segment_length):
    segments = []
    for start_time in range(0, len(audio), segment_length):
        end_time = start_time + segment_length
        segment = audio[start_time:end_time]
        segments.append(segment)
    return segments

def get_text_from_segment(segment):
    file = temp_directory + "/segmenttemp.mp3"
    if os.path.exists(file):
        os.remove(file)
    segment.export("./" + file, format="mp3")
    segment_file = open(file, "rb")
    response = openai.Audio.transcribe(
        model = "whisper-1", 
        file = segment_file,
        response_format = "json",
        language = "en",
    )
    if os.path.exists(file):
        os.remove(file)
    return(response["text"])

def create_audio_text(file_name):
    create_temp_directory()
    audio_file = AudioSegment.from_file(os.path.join(sony_directory, file_name))
    segments = break_audio_into_segments(audio_file, segment_time)
    response_text = ""
    for i, segment in enumerate(segments):
        response_text += get_text_from_segment(segment)
    return(response_text)

def get_summary_for_file(file_name):
    """
    Produces summary text from the begining of the audio file to provide more context to the file name
    The file text is limited to 200 characters
    """
    create_temp_directory()
    filelocation = os.path.join(sony_directory, file_name)
    audio_file = AudioSegment.from_file(filelocation)
    segment = audio_file[0:summary_time]
    # Temp File
    file = temp_directory + "/tempsummary.mp3"
    if os.path.exists(file):
        os.remove(file)
    segment.export("./" + file, format="mp3")
    summary_file = open(file, "rb")
    # Transcribe the first 15 seconds
    response = openai.Audio.transcribe(
        model = "whisper-1", 
        file = summary_file,
        response_format = "json",
        language = "en",
    )
    if len(str(summary_text_for_filename(response["text"]))) > 200:
        return(str(summary_text_for_filename(response["text"]))[0:200])
    else:
        return(str(summary_text_for_filename(response["text"])))

def check_move_audio_file_directory():
    if not os.path.exists(audiobackup_directory):
        os.makedirs(audiobackup_directory)

def move_audio_file(file_name):
    check_move_audio_file_directory()
    orig_file_path = os.path.join(sony_directory, file_name)
    dest_file_path = os.path.join(audiobackup_directory, file_name)
    shutil.copy(orig_file_path, dest_file_path)

def check_move_text_file_directory():
    if not os.path.exists(textcreation_directory):
        os.makedirs(textcreation_directory)

def delete_audio_file(file_name):
    os.remove(os.path.join(sony_directory, file_name))

def create_text_file(text, file_name):
    check_move_text_file_directory()
    file_path = os.path.join(textcreation_directory, file_name)
    with open(file_path, "w") as text_file:
        text_file.write(text)
        text_file.close()

def parse_audio_file(audio_file_name):
    """
    Parse the audio file name to get the date and time into components
    """
    date, time = audio_file_name.removesuffix(".mp3").split("_")
    year = date[0:2]
    month = date[2:4]
    day = date[4:6]
    hour = time[0:2]
    minute = time[2:4]
    return date, time, year, month, day, hour, minute

def main_processing():
    """
    The orchistation of the processing of the files
    """
    # Checking for directory of Sony created files 
    if not check_sony_file_directory():
        print("Sony directory not present to process")
        exit()

    # Getting files from Sony
    file_list = get_files_from_sony()
    print(f"Number of files: {len(file_list)}")
    
    # Loop through files
    for file_name in file_list:
        # Generate File Information
        date, time, year, month, day, hour, minute = parse_audio_file(str(file_name))
        
        print(f"Processing file: {file_name}")

        # Just in case we still have spaces
        text_file_name = date + "_" + time + "_" + str(summary_text_for_filename(str(get_summary_for_file(file_name))).replace(" ", "_")) + ".txt"

        # Generate Audio Text
        audio_text = create_audio_text(file_name)

        # Produce Text File
        create_text_file(audio_text, text_file_name)

        # Move Audio File
        move_audio_file(file_name)

        # Delete Audio File
        delete_audio_file(file_name)
        
        # Final Processing Message
        print(f"File: {file_name} processed")
    
if (True):
    main_processing()
else:
    print("Not handling files at this time")