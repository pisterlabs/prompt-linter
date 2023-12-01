import os
import openai
from dotenv import load_dotenv
from music21 import *
from midi2audio import FluidSynth

load_dotenv()

class Skitz:
    def __init__(self):
        # Load OpenAI API key from environment variables
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if openai.api_key is None:
            raise ValueError("OpenAI API key not found in environment variables")

        # Check if directories exist, if not, create them
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.documentation_path = os.path.join(self.base_path, 'Documentations')
        self.inspiration_path = os.path.join(self.base_path, 'Inspirations')
        self.generations_path = os.path.join(self.base_path, 'Generations')
        self.log_path = os.path.join(self.base_path, 'Logs')

        os.makedirs(self.documentation_path, exist_ok=True)
        os.makedirs(self.inspiration_path, exist_ok=True)
        os.makedirs(self.generations_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        # Read instructions from file
        try:
            with open(os.path.join(self.documentation_path, 'instructions.md'), 'r') as file:
                self.instructions = file.read()
        except FileNotFoundError:
            raise FileNotFoundError("Instructions file not found")

        # Load inspiration content
        self.inspiration = self.load_inspiration()

        # Initialize total tokens used
        self.total_tokens = 0

    def load_inspiration(self):
        # Load inspiration from file if it exists, otherwise return an empty string
        try:
            with open(os.path.join(self.inspiration_path, 'inspiration.md'), 'r') as file:
                inspiration = file.read()
                return inspiration
        except FileNotFoundError:
            print("The inspiration file is not found, continuing with the program...")
            return ""
        
    

    def extract_markdown(self, song):
        # Split the song into lines
        lines = song.split('\n')

        # Find the start and end of the markdown content
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.strip() == '```':
                if start is None:
                    start = i
                else:
                    end = i
                    break

        # Extract the markdown content
        if start is not None and end is not None:
            markdown = '\n'.join(lines[start+1:end])
        else:
            markdown = song

        return markdown

    def quality_check(self, song, user_instructions):
        prompt = f"Check this output and see if it is in the correct ABC syntax. If not, change it so that it is. Here is the output: \n{song}"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant. You check and correct the format/syntax of a song. Here is the ABC documentation:\n{self.instructions} & here is the user's instructions:\n{user_instructions}. Make sure the users instructions are being followed."},
                    {"role": "user", "content": prompt},
                ]
            )
            self.total_tokens += response['usage']['total_tokens']
            self.log_messages(prompt, response.choices[0].message['content'])
        except openai.OpenAIError as e:
            print(f"\nAn error occurred: {e}")
            return None

        corrected_song = response.choices[0].message['content']

        return corrected_song

    def compose_song(self, user_instructions):
        prompt = f"# ABC Player Specification\n\nUse these instructions to complete the request only respond with ABC format:\n\n{user_instructions}\n\nInspiration:\n{self.inspiration}"

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. You can generate creative and original music based on the input requirements given to you."},
                    {"role": "user", "content": prompt},
                ]
            )
            self.total_tokens += response['usage']['total_tokens']
            self.log_messages(prompt, response.choices[0].message['content'])
        except openai.OpenAIError as e:
            print(f"\nAn error occurred: {e}")
            return None

        song = response.choices[0].message['content']

        # Ensure the song has a default note length
        if "L:" not in song:
            song = "L:1/8\n" + song

        # Extract the markdown content
        song = self.extract_markdown(song)

        # Perform a quality check on the output song by default
        corrected_song = self.quality_check(song, user_instructions)
        return corrected_song

    def write_abc_file(self, song, filename):
        # Determine full directory path without the filename
        dir_path = os.path.join(self.generations_path, "/".join(filename.split("/")[:-1]))

        # Create the directories if they do not exist
        os.makedirs(dir_path, exist_ok=True)

        # Prepare the filename, ensuring uniqueness
        original_filename = filename.split("/")[-1]
        counter = 2
        while os.path.exists(os.path.join(dir_path, original_filename)):
            original_filename = f"{original_filename[:-4]}_{counter}.abc"
            counter += 1

        # Determine full file path
        filepath = os.path.join(dir_path, original_filename)

        # Write the song to the file
        with open(filepath, 'w') as file:
            file.write(song)

        print(f"File saved as: {filepath}")
        return filepath



    def convert_to_midi(self, abc_file, midi_file):
        # Convert ABC to MIDI using music21
        abcScore = converter.parse(abc_file)
        midiScore = abcScore.write('midi', midi_file)

    def convert_to_audio(self, midi_file, audio_file):
        # Convert MIDI to audio using FluidSynth
        fs = FluidSynth()
        fs.midi_to_audio(midi_file, audio_file)

    def get_user_input_easy(self):
        user_instructions = ""
        questions = [
            "\nPlease enter the genre of the song: ",
            "\nPlease enter the tempo of the song: ",
            "\nPlease enter any specific lyrics or themes you'd like to include: ",
            "\nPlease enter the desired chord progression: ",
            "\nPlease enter any additional instructions or preferences: ",
            "\nPlease enter the desired length of the song (short, medium, long): ",
            "\nDo you want the song to have a specific structure (verse, chorus, bridge, etc.)? If so, please specify: ",
            "\nDo you want the song to have a specific mood or emotion? If so, please specify: ",
            "\nDo you want the song to tell a story? If so, please briefly describe the story: ",
             "\nDo you want to perform a quality check on the output? (yes/no): ",
        ]

        for question in questions:
            answer = input(question)
            user_instructions += answer + "\n"

        return user_instructions.strip()

    def get_user_input_advanced(self):
        user_instructions_advanced = ""
        questions = [
            "\nPlease enter the genre of the song: ",
            "\nPlease enter the tempo of the song: ",
            "\nPlease enter any specific lyrics or themes you'd like to include: ",
            "\nPlease enter the desired chord progression: ",
            "\nPlease enter any additional instructions or preferences: ",
            "\nPlease enter the desired length of the song (short, medium, long): ",
            "\nDo you want the song to have a specific structure (verse, chorus, bridge, etc.)? If so, please specify: ",
            "\nDo you want the song to have a specific mood or emotion? If so, please specify: ",
            "\nDo you want the song to tell a story? If so, please briefly describe the story: ",
            "\nPlease enter the key of the song: ",
            "\nPlease enter the meter of the song: ",
            "\nPlease enter the default length of a note: ",
            "\nPlease enter the composer of the song: ",
            "\nPlease enter the title of the song: ",
            "\nDo you want to perform a quality check on the output? Recomended (HIGHER API COST)(yes/no): ",
        ]

        for question in questions:
            answer = input(question)
            user_instructions_advanced += answer + "\n"

        return user_instructions_advanced.strip()

    def log_messages(self, input_message, output_message):
        # Create a new log file for each song
        log_file = os.path.join(self.log_path, f'log_{self.total_tokens}.txt')
        with open(log_file, 'w') as file:
            file.write(f'Input: {input_message}\nOutput: {output_message}')

    def generate_song(self):
        print("\n")
        print("""
                █████████████████████████████████████████████████████
                █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
                █░░░▄▄▄░░░░░▄▄▄▄▄▄░░░░▄▄▄░░▄▄▄░░░░░░░▄▄░░░░▄▄▄░░░█
                █░░░███░░░░██░░░░██░░░███░░███░░░░░░░███░░░███░░░█
                █░░░███░░░░██░░▄▄▄██░░░███░░▄▄▄░░░░░░░███░░░███░░░█
                █░░░███░░░░██░░░░██░░░███░░░░██░░░░░░░███░░░░░░░░█
                █░░░▀▀▀░░░░▀▀▀▀▀▀░░░░▀▀▀░░░░░▀▀▀░░░░░░░▀▀░░░░▀▀▀░░█
                █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░█
                █████████████████████████████████████████████████████

        """)
        print("\nWelcome to Skitz! This program will generate a song for you based on your input.")
        print("Please note that this program is still in development and may not always work as intended.")
        print("Please also note that this program uses the OpenAI API, which is a paid service. You will be charged for each request.")
        print("If vscode wont run abc code, please visit to run online: https://abc.rectanglered.com")
        
        mode = input("Please select a mode: \n1. Easy \n2. Advanced\n")
        
        if mode == '1':
            user_instructions = self.get_user_input_easy()
        elif mode == '2':
            user_instructions = self.get_user_input_advanced()
        else:
            print("Invalid mode selected. Please try again.")
            return

        filepath = self.generate_song_with_same_instructions(user_instructions)

        if filepath:
            while True:
                regenerate = input("\nDo you want to regenerate the song with the same instructions? (yes/no): ")
                if regenerate.lower() == 'yes':
                    filepath = self.generate_song_with_same_instructions(user_instructions)
                elif regenerate.lower() == 'no':
                    break
                else:
                    print("Invalid input. Please type 'yes' or 'no'.")
        else:
            print("\nFailed to generate a song. Please try again.")

    def generate_song_with_same_instructions(self, user_instructions):
        print("\nGenerating your song. This may take a few moments...")
        song = self.compose_song(user_instructions)

        if song is None:
            print("\nFailed to generate a song. Please try again.")
            return

        # Extract the title from the user instructions
        lines = user_instructions.split('\n')
        title_line = next((line for line in lines if line.startswith("Please enter the title of the song:")), None)

        if title_line is not None:
            # Extract the title from the line
            title = title_line.split(":")[1].strip()

            # Use the title as the filename (replacing spaces with underscores)
            filename = f"{title.replace(' ', '_')}.abc"
        else:
            # If no title was found, fall back to the original filename
            filename = f"{user_instructions[:20].replace(' ', '_')}.abc"

        filepath = self.write_abc_file(song, filename)

        if filepath:
            print(f"\nThe song has been written to: {filepath}")
        else:
            print("\nFailed to write the song to a file. Please try again.")

        return filepath


if __name__ == "__main__":
    try:
        skitz = Skitz()
        skitz.generate_song()
    except Exception as e:
        print(f"An error occurred: {e}")
