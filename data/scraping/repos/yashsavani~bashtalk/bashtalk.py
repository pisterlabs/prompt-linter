import platform
import argparse
import re
import subprocess
import openai
import os
import llm
import tempfile


def record_voice(file_path):
    """Records voice using sox and saves to a temporary file."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        print("Recording. Press Ctrl+C to stop.")

        sox_process = subprocess.Popen(["sox", "-q", "-d", temp_wav_path])
        sox_process.wait()

    except KeyboardInterrupt:
        sox_process.terminate()
        sox_process.wait()
        print("Recording stopped.")

    subprocess.run(["sox", temp_wav_path, file_path])
    os.remove(temp_wav_path)


def convert_to_text(file_path):
    """Converts audio file to text using OpenAI's Whisper ASR."""
    with open(file_path, "rb") as f:
        transcription = openai.Audio.transcribe("whisper-1", f)
    return transcription["text"]


def get_system_info():
    uname_info = platform.uname()
    if uname_info.system == 'Darwin':
        mac_info = platform.mac_ver()
        uname_info = f"macOS {mac_info[0]} ({mac_info[2]})"
    else:
        uname_info = uname_info.system

    username = os.getlogin()
    shell = os.environ.get("SHELL", "Unknown")
    curr_dir = os.getcwd()
    ls_output = subprocess.check_output("ls", text=True).strip()

    system_info_str = f"OS: {uname_info},\n" \
                      f"User: {username},\n" \
                      f"Shell: {shell},\n" \
                      f"Current Directory: {curr_dir},\n" \
                      f"Output of ls: \n{ls_output}"
    return system_info_str


def read_files(file_paths):
    file_contents = {}
    for path in file_paths:
        with open(path, 'r') as f:
            file_contents[os.path.basename(path)] = f.read()
    return file_contents


def run_llm(text, model, context_file_paths=None):
    sys_info = get_system_info()

    context_prompt = ''
    if context_file_paths:
        file_contents = read_files(context_file_paths)
        context_prompt += "Context Files:\n"
        for file_name, content in file_contents.items():
            # Limit content length
            context_prompt += f"{file_name}: {content[:1000]}\n"

    prompt = (f"System Info:\n{sys_info}\n\n"
              f"{context_prompt}\n\n"
              "Give me a shell script inside ```bash ``` to do the following:\n\n"
              "You can utilize the cli command `llm -m 4 <prompt>` in the bash script to run an large language model with a prompt if needed.\n"
              "You can utilize the GNU parallel command to run multiple commands in parallel.\n"
              f"{text}")

    return model.prompt(prompt).text()


def extract_script(text):
    """Extracts the bash script from llm output."""
    pattern = r'```bash\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 0:
        return [""]
    return matches


def main():
    parser = argparse.ArgumentParser(description="bashtalk")
    parser.add_argument("-d", "--dir_path", default="/tmp",
                        help="Directory to save files")
    parser.add_argument(
        "-f", "--file_name", default="audio_record.mp3", help="Name of the audio file")
    parser.add_argument(
        "-m", "--model_name", default="gpt-3.5-turbo", help="LLM model name")
    parser.add_argument("-s", "--skip_llm", action="store_true",
                        help="Skip running the LLM model")
    parser.add_argument("-y", "--auto_run", action="store_true",
                        help="Automatically run the script without asking")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("-c", "--context_files", nargs="*", default=[],
                        help="Context files for LLM prompt")

    args = parser.parse_args()
    file_path = os.path.join(args.dir_path, args.file_name)

    record_voice(file_path)

    if args.verbose:
        print()
        print("Converting audio to text...")

    transcribed_text = convert_to_text(file_path)

    if args.verbose:
        print("Transcribed text:")
        print(transcribed_text)
        print()

    if not args.skip_llm:
        model = llm.get_model(args.model_name)
        context_file_paths = args.context_files
        llm_output = run_llm(transcribed_text, model, context_file_paths)

        if args.verbose:
            print("LLM output:")
            print(llm_output)
            print()

        script_text = extract_script(llm_output)[0]

        script_file_path = os.path.join(args.dir_path, "bash_script.sh")
        with open(script_file_path, "w") as f:
            f.write(script_text)

        if args.verbose:
            subprocess.run(["bat", script_file_path])
            # print("Script:")
            # print(script_text)

        if not args.auto_run:
            print("Do you want to run the script? [Y/n]")
            answer = input().strip().lower()
            if answer != "n":
                subprocess.run(["bash", script_file_path])
        else:
            subprocess.run(["bash", script_file_path])

if __name__ == '__main__':
    main()
