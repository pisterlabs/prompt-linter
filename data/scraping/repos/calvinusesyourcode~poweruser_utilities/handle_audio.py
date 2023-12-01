import os, openai, subprocess, pathlib, math, datetime, json, inquirer, time, whisper
from pathlib import Path
# my imports
from handle_strings import to_time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_audio_size(filename: pathlib.WindowsPath):
    """Get the size of an audio file in MB."""
    cmd = ["ffprobe", '-i', filename, '-show_entries', 'format=size', '-v', 'quiet', '-of', 'csv=p=0']
    output = subprocess.check_output(cmd)
    return int(output)/1024/1024

def get_audio_duration(filename: pathlib.WindowsPath):
    """Get the duration of an audio file."""
    cmd = ['ffprobe', '-i', filename, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
    output = subprocess.check_output(cmd)
    return float(output)

def downformat_audio(folder,filename):
    """Reformat an audio file to 16kHz mono."""
    output_file = Path(folder,Path(filename).stem+"_reformatted.mp3")
    command = ["ffmpeg", "-i", Path(folder,filename), "-vn", "-ac", "1", "-ar", "16000", "-ab", "192k", "-y", output_file]
    subprocess.run(command, check=True)
    return Path(output_file)

def transcribe(filepath):
    return whisper.load_model("tiny").transcribe(filepath)["text"]

def split_audio_file(folder,filename):
    """Split an audio file into 24MB chunks, especially for use with Whisper API."""
    files = []
    filename = Path(folder,filename)
    output_title_stem = Path(filename).stem
    total_duration = get_audio_duration(filename)
    chunk_duration = int(get_audio_duration(filename)/(get_audio_size(filename)/9))
    num_chunks = math.ceil(total_duration / chunk_duration)
    
    for i in range(num_chunks):
        overlap = 2
        start_time = to_time(i * chunk_duration) if i == 0 else to_time((i * chunk_duration) - overlap)
        end_time = to_time(((i+1) * chunk_duration) + overlap)
        output = Path(f'{folder}/{output_title_stem}_{i}.mp3')
        cmd = ['ffmpeg', '-i', filename, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy', output]
        subprocess.run(cmd, check=True)
        files.append(Path(output))
    return files

def audio_mp4_to_mp3(folder, filename: str):
    """Convert an mp4 audio file to mp3."""
    
    file_stem = Path(filename).stem
    mp4 = str(Path(folder, file_stem + ".mp4"))
    mp3 = str(Path(folder, file_stem + ".mp3"))
    command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', mp4]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    info = json.loads(result.stdout)

    # Extract the audio codec and sample rate from the info
    audio_info = [stream for stream in info['streams']][0]
    sample_rate = audio_info['sample_rate']

    command = ['ffmpeg', '-i', mp4, '-codec:a', 'libmp3lame', '-ar', sample_rate, mp3]

    print(" ".join(command))
    result = subprocess.run(command, shell=True, check=True)

    # Check the result
    if result.returncode != 0:
        raise Exception(f'An error occurred: {result.returncode}')
    else:
        mp3_size = os.path.getsize(mp3)
        mp4_size = os.path.getsize(mp4)
        if abs(mp3_size - mp4_size) < 0.1 * mp4_size:
            os.remove(mp4)
        return Path(mp3)

def trim_audio(folder, filename, start_time:str="00:00:00", end_time:str="end"):
    """Trim an audio file."""
    noFilename = True
    i = 0
    while noFilename:
        output_filename = Path(Path(filename).stem+f"_{i}.mp3")
        if not os.path.exists(Path(folder,output_filename)):
            noFilename = False
        i += 1

    if end_time == "end":
        end_time = to_time(get_audio_duration(Path(folder,filename)))

    file_path = str(Path(folder,filename))
    output_path = str(Path(folder,output_filename))
    command = ["ffmpeg", "-i", file_path, "-ss", start_time, "-to", end_time, "-c", "copy", "-y", output_path]
    subprocess.run(command, check=True)
    return output_path

def trim_audio_with_ui():
    folder = "downloads"
    files = os.listdir(folder)

    audio_files = [file for file in files]
    if not audio_files:
        print("No audio files found in the downloads folder.")
        time.sleep(5)
        return

    questions = [
        inquirer.List(
            "file",
            message="Select a file to trim",
            choices=audio_files
        ),
        inquirer.Text(
            "start_time",
            message="Enter the start time (format HH:MM:SS, default is 00:00:00)"
        ),
        inquirer.Text(
            "end_time",
            message="Enter the end time (format HH:MM:SS, default is 'end')"
        ),
    ]

    answers = inquirer.prompt(questions)
    start_time = answers["start_time"] if answers["start_time"] else "00:00:00"
    end_time = answers["end_time"] if answers["end_time"] else "end"
    trim_audio(folder, answers["file"], start_time, end_time)

    subprocess.Popen(f'explorer "{folder}"')

def audio_to_mp3(folder, filename: str):
    """Convert any audio file to mp3."""

    # Skip the process if the file is already an mp3
    if filename.endswith('.mp3'):
        print(f"{filename} is already an mp3 file. Skipping conversion.")
        return Path(folder, filename)

    file_stem = Path(filename).stem
    file_ext = Path(filename).suffix
    input_file = str(Path(folder, file_stem + file_ext))
    output_file = str(Path(folder, file_stem + ".mp3"))

    command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', input_file]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    info = json.loads(result.stdout)

    # Extract the audio codec and sample rate from the info
    audio_info = [stream for stream in info['streams'] if stream['codec_type'] == 'audio'][0]
    sample_rate = audio_info['sample_rate']

    # Now convert the file
    command = ['ffmpeg', '-i', input_file, '-codec:a', 'libmp3lame', '-ar', sample_rate, output_file]
    print(" ".join(command))
    result = subprocess.run(command, shell=True, check=True)

    # Check the result
    if result.returncode != 0:
        raise Exception(f'An error occurred: {result.returncode}')
    else:
        output_size = os.path.getsize(output_file)
        input_size = os.path.getsize(input_file)
        if abs(output_size - input_size) < 0.1 * input_size:
            os.remove(input_file)
            return Path(output_file)

def to_mp3_with_ui():
    folder = "downloads"
    files = os.listdir(folder)

    audio_files = [file for file in files if not file.endswith('.mp3')]  # changed from .mp3 to not .mp3
    if not audio_files:
        print("No non-mp3 audio files found in the downloads folder.")
        time.sleep(5)
        return

    questions = [
        inquirer.List(
            "file",
            message="Select a file to convert to mp3",
            choices=audio_files
        ),
    ]

    answers = inquirer.prompt(questions)
    audio_to_mp3(folder, answers["file"])

    subprocess.Popen(f'explorer "{folder}"')

def audio_to_audio(folder, filename: str, target_format: str, target_sample_rate: int = None):
    """Convert any audio file to a specified format."""

    codec_map = {
        'mp3': 'libmp3lame',
        'aac': 'libvo_aacenc',
        'm4a': 'libvo_aacenc',  # Add this line
        'ac3': 'ac3',
        'flac': 'flac',
        'wav': 'pcm_s16le',
        'ogg': 'libvorbis',
        'wma': 'wmav2',
        'opus': 'libopus',
    }

    file_stem = Path(filename).stem
    file_ext = Path(filename).suffix.lstrip('.')  # get extension without the dot

    # Skip the process if the file is already in the target format
    if file_ext.lower() == target_format.lower():
        print(f"{filename} is already a {target_format} file. Skipping conversion.")
        return Path(folder, filename)

    # Ensure we have a codec for the target format
    if target_format.lower() not in codec_map:
        print(f"Unsupported target format: {target_format}")
        return

    input_file = str(Path(folder, file_stem + '.' + file_ext))
    output_file = str(Path(folder, file_stem + '.' + target_format))

    command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', input_file]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    info = json.loads(result.stdout)

    # Extract the audio codec and sample rate from the info
    audio_info = [stream for stream in info['streams'] if stream['codec_type'] == 'audio'][0]
    sample_rate = audio_info['sample_rate'] if target_sample_rate is None else str(target_sample_rate)

    # Now convert the file
    command = ['ffmpeg', '-i', input_file, '-codec:a', codec_map[target_format.lower()], '-ar', sample_rate, output_file]
    print(" ".join(command))
    result = subprocess.run(command, shell=True, check=True)

    # Check the result
    if result.returncode != 0:
        raise Exception(f'An error occurred: {result.returncode}')
    else:
        output_size = os.path.getsize(output_file)
        input_size = os.path.getsize(input_file)
        if abs(output_size - input_size) < 0.1 * input_size:
            os.remove(input_file)
        return Path(output_file)

def audio_to_audio_with_ui():
    folder = "downloads"
    files = os.listdir(folder)

    audio_files = [file for file in files if '.' in file]  # get all files with an extension
    if not audio_files:
        print("No audio files found in the downloads folder.")
        time.sleep(5)
        return

    codec_map = {
        'mp3': 'libmp3lame',
        'aac': 'libvo_aacenc',
        'm4a': 'libvo_aacenc',  # Add this line
        'ac3': 'ac3',
        'flac': 'flac',
        'wav': 'pcm_s16le',
        'ogg': 'libvorbis',
        'wma': 'wmav2',
        'opus': 'libopus',
    }

    target_formats = list(codec_map.keys())

    questions = [
        inquirer.List(
            "file",
            message="Select a file to convert",
            choices=audio_files
        ),
        inquirer.List(
            "target_format",
            message="Select the target format",
            choices=target_formats
        ),
        inquirer.Text(
            "target_sample_rate",
            message="Sample rate"
        ),
    ]

    answers = inquirer.prompt(questions)
    target_sample_rate = int(answers["target_sample_rate"]) if answers["target_sample_rate"] else None
    audio_to_audio(folder, answers["file"], answers["target_format"], target_sample_rate)

    subprocess.Popen(f'explorer "{folder}"')

def downsample_all(folder: str, extensions: list, sample_rate: int):
    from pathlib import Path
    """Downsample all audio files in a folder and its subfolders."""
    # Create a new folder with the same name but appended with "_sampled"
    new_folder = Path(folder).parent / (Path(folder).stem + "_sampled")
    os.makedirs(new_folder, exist_ok=True)

    # Walk through the folder and its subfolders
    for dirpath, dirnames, filenames in os.walk(folder):
        # Create corresponding subfolders in the new folder
        relative_dirpath = Path(dirpath).relative_to(folder)
        new_dirpath = new_folder / relative_dirpath
        os.makedirs(new_dirpath, exist_ok=True)

        # Process each file
        for filename in filenames:
            # Check if the file has one of the specified extensions
            if Path(filename).suffix.lstrip('.') in extensions:
                input_file = str(Path(dirpath, filename))
                output_file = str(new_dirpath / filename)
                downsample_and_move(input_file, output_file, sample_rate)

def downsample_and_move(input_file: str, output_file: str, sample_rate: int):
    """Downsample an audio file and move it to a specified location."""
    # Downsample the file
    command = ['ffmpeg', '-i', input_file, '-ar', str(sample_rate), output_file]
    subprocess.run(command, check=True)

def downsample_all_with_ui():
    """Interactively downsample all audio files in a folder and its subfolders."""
    import inquirer

    questions = [
        inquirer.Text("folder", message="Enter the folder path"),
        inquirer.Text("extensions", message="Enter the file extensions, separated by commas"),
        inquirer.Text("sample_rate", message="Enter the desired sample rate"),
    ]

    answers = inquirer.prompt(questions)
    folder = answers["folder"]
    extensions = [ext.strip() for ext in answers["extensions"].split(',')]
    sample_rate = int(answers["sample_rate"])

    downsample_all(folder, extensions, sample_rate)

def whisper_test(file_path):
    from openai import OpenAI
    client = OpenAI()

    audio_file= open(file_path, "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    response_format="verbose_json",
    )
    # for each attribute of transcript, save the key and value to file
    # save transcript to txt file
    with open("transcript.txt", "w") as f:
        f.write(json.dumps(transcript.text, indent=4))

# whisper_test("C:/Users/calvi/3D Objects/poweruser_utilities/downloads/audio_____alcohol_is_worse_than_you_think__andrew_huberman.mp3")
# whisper_test("C:/Users/calvi/3D Objects/poweruser_utilities/downloads/audio_____somebody_ring_the_dinkster.mp4")