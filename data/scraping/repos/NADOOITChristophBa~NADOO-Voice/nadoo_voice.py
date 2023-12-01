import os
from pathlib import Path
from dotenv import load_dotenv
import openai
import tkinter as tk
from tkinter import simpledialog, ttk, scrolledtext
import json
import re
import threading
import time
import tkinter.filedialog as filedialog


import os
from pathlib import Path

# Function to convert text to speech and save as an MP3 file
import os
from pathlib import Path


def parse_config_matrix(config_str, total_chapters):
    if not config_str:
        return {str(chapter).zfill(2): True for chapter in range(1, total_chapters + 1)}

    chapters_config = {}
    for part in config_str.split(","):
        if "-" in part:
            start, end = part.split("-")
            start = int(start)
            end = total_chapters if end == "*" else int(end)
            for chapter in range(start, end + 1):
                chapters_config[str(chapter).zfill(2)] = True
        elif "+" in part:
            chapters = part.split("+")
            for chapter in chapters:
                chapters_config[chapter.zfill(2)] = True
        else:
            chapters_config[part.zfill(2)] = True
    return chapters_config


from pathlib import Path
import os
import re


def get_audio_file_path_for_chapter_info(book_title, chapter_title, voice, output_file):
    """
    Generates the file path for an audio file based on book title, chapter title, voice, and output file name.

    Parameters:
    - book_title (str): The title of the book.
    - chapter_title (str): The title of the chapter.
    - voice (str): The voice used for TTS.
    - output_file (str): The name of the output audio file.

    Returns:
    - str: The full path for the audio file.
    """
    # Sanitize book and chapter titles to use in file paths
    sanitized_book_title = re.sub(r'[<>:"/\\|?*]', "_", book_title)
    sanitized_chapter_title = re.sub(r'[<>:"/\\|?*]', "_", chapter_title)

    # Create the folder structure
    base_folder = (
        Path(__file__).parent / sanitized_book_title / voice / sanitized_chapter_title
    )
    os.makedirs(base_folder, exist_ok=True)

    # Return the modified output file path
    return f"{base_folder}/{output_file}"


# Function to convert text to speech and save as an MP3 file
def text_to_speech(
    input_text,
    model="tts-1-hd",
    voice="onyx",
    output_file="speech.mp3",
    book_title="Untitled",
    chapter_title="Chapter",
):
    retry_count = 0
    retry_delay = 10  # Initial delay in seconds

    while True:  # Infinite loop, will break on success or non-rate-limit error
        try:
            # Generate the full path for the audio file
            modified_output_file = get_audio_file_path_for_chapter_info(
                book_title, chapter_title, voice, output_file
            )

            client = openai.OpenAI()

            # Create the spoken audio from the input text
            response = client.audio.speech.create(
                model=model, voice=voice, input=input_text
            )

            # Stream the response to the file
            response.stream_to_file(Path(modified_output_file))
            print(f"Audio file saved as {modified_output_file}")
            break  # Break the loop if successful

        except Exception as e:
            error_message = str(e)
            if "rate_limit_exceeded" in error_message:
                print(f"Rate limit reached, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay = min(
                    retry_delay * 2, 1200
                )  # Double the delay each time, max 20 minutes
                retry_count += 1
            else:
                print(f"An error occurred: {error_message}")
                break

    if retry_count > 0:
        print(f"Retried {retry_count} times before success.")


def get_chapter_audio_for_chapter(chapter, chapter_number, voice, model, book_title):
    chapter_audio_data = []
    title = chapter.get("chapter_title", "Untitled")
    text = chapter.get("chapter_content", "")

    print(f"Processing chapter: {title}")
    print(f"Chapter number: {chapter_number}")

    # Decide whether to split into subchapters
    should_split = len(text) > 4000
    subchapters = split_into_subchapters(text) if should_split else [text]

    for i, subchapter_content in enumerate(subchapters, start=1):
        combined_text = (
            f"{title} Teil {i}. {subchapter_content}"
            if len(subchapters) > 1
            else f"{title}. {subchapter_content}"
        )
        sanitized_title = sanitize_filename_from_title(title, chapter_number, i)

        audio_path = text_to_speech(
            input_text=combined_text,
            book_title=book_title,
            model=model,
            voice=voice,
            output_file=sanitized_title + ".mp3",
        )

        chapter_audio = {"text": combined_text, "audio_path": audio_path}
        chapter_audio_data.append(chapter_audio)

    return chapter_audio_data


def get_default_voice_model_matrix(default_chapters, predefined_matrix=None):
    """
    Generates a voice-model matrix, using a predefined matrix if provided,
    or creates a default matrix based on the default chapters.

    Parameters:
    - default_chapters (str): A string representing the default chapters to be processed.
      Can be a range (e.g., "1-10"), a list of chapters (e.g., "1,3,5"), or "*" for all chapters.
    - predefined_matrix (dict, optional): A predefined nested dictionary mapping voices to models
      and their respective chapters. If provided, this matrix is used as is.

    Returns:
    - dict: A nested dictionary where each key is a voice, mapping to another dictionary
      of models and their respective chapter specifications.

    Example Usage:
    - get_default_voice_model_matrix("*") -> processes all chapters for each voice-model combination.
    - get_default_voice_model_matrix("1-10") -> processes chapters 1 to 10 for each voice-model combination.
    - get_default_voice_model_matrix("*", predefined_matrix=my_predefined_matrix)
      -> uses the predefined matrix directly.
    """
    """ 
    predefined_matrix = {
    "alloy": {
        "tts-1": "1-5",
        "tts-1-hd": "6-10"
    },
    "echo": {
        "tts-1-f": "11-15"
    }
    # ... other configurations ...
    }
    """
    if predefined_matrix:
        return predefined_matrix

    # List of available voices
    available_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    # List of available models
    # available_models = ["tts-1", "tts-1-f", "tts-1-m", "tts-1-hd", "tts-1-hd-f"]
    available_models = ["tts-1-hd"]

    # Creating the default matrix if no predefined matrix is provided
    return {
        voice: {model: default_chapters for model in available_models}
        for voice in available_voices
    }


def check_audio_files_existence(chapters, book_title, voice_model_matrix):
    """
    Checks if the audio files for each chapter were created successfully.

    Parameters:
    - chapters (list): List of chapters.
    - book_title (str): The title of the book.
    - voice_model_matrix (dict): A matrix mapping voices to models and their respective chapters.
    """
    missing_files = []

    for voice, models in voice_model_matrix.items():
        for model, chapter_selection in models.items():
            chapters_to_process = parse_chapter_selection(
                chapter_selection, len(chapters)
            )

            for chapter_number in chapters_to_process:
                # Generate the expected audio file path
                chapter_title = f"Chapter_{chapter_number}"
                expected_file_path = get_audio_file_path_for_chapter_info(
                    book_title, chapter_title, voice, f"{chapter_number}.mp3"
                )

                # Check if the file exists
                if not os.path.exists(expected_file_path):
                    missing_files.append(expected_file_path)

    if missing_files:
        print("Warning: The following audio files were not created successfully:")
        for missing_file in missing_files:
            print(missing_file)
    else:
        print("All audio files created successfully.")


def get_chapters_audio_for_chapters(
    chapters, book_title, voice_model_matrix=None, default_chapters="*"
):
    chapter_audios = []
    if voice_model_matrix is None:
        voice_model_matrix = get_default_voice_model_matrix(default_chapters)

    for voice, models in voice_model_matrix.items():
        for model, chapter_selection in models.items():
            chapters_to_process = parse_chapter_selection(
                chapter_selection, len(chapters)
            )

            for chapter_number in chapters_to_process:
                print(f"Processing {voice} {model} for Chapter {chapter_number}")
                # Directly calling get_chapter_audio_for_chapter
                chapter_audio_data = get_chapter_audio_for_chapter(
                    chapters[chapter_number - 1],
                    chapter_number,
                    voice,
                    model,
                    book_title,
                )
                chapter_audios.extend(chapter_audio_data)

    # Call the updated check_audio_files_existence function
    check_audio_files_existence(chapters, book_title, voice_model_matrix)
    return chapter_audios


def get_chapters_audio_for_chapters_01(
    chapters, book_title, voice_model_matrix="", default_chapters="*"
):
    chapter_audios = []
    available_voices = ["alloy", "echo", "fable", "nova", "shimmer"]
    available_models = ["tts-1", "tts-1-f", "tts-1-m", "tts-1-hd", "tts-1-hd-f"]

    # Example voice-model matrix
    voice_model_matrix = {
        "alloy_tts-1-hd": "02,08,12,14,22,25,36,39,42,57",
        "echo_tts-1-hd": "02,08,12,14,22,25,36,38,39,42,57",
        "fable_tts-1-hd": "07,13,38,39,42,57",
        "nova_tts-1-hd": "14,24,38,41,48",
        "shimmer_tts-1-hd": "07,13,14,24,37,41,44",
    }

    # Use default_chapters for all voice-model combinations if matrix is not provided
    if not voice_model_matrix:
        voice_model_matrix = {
            f"{voice}_{model}": default_chapters
            for voice in available_voices
            for model in available_models
        }

    for voice_model_key, chapter_selection in voice_model_matrix.items():
        voice, model = voice_model_key.split("_")
        chapters_to_process = parse_chapter_selection(chapter_selection, len(chapters))

        for chapter_number in chapters_to_process:
            print(f"Processing {voice} {model} for Chapter {chapter_number}")
            chapter_audio_data = get_chapter_audio_for_chapter(
                chapters[chapter_number - 1], chapter_number, voice, model, book_title
            )
            chapter_audios.extend(chapter_audio_data)

    return chapter_audios


def parse_chapter_selection(chapter_selection, total_chapters):
    """
    Parse the chapter selection string to a list of chapter numbers.
    """
    chapter_numbers = []
    for part in chapter_selection.split(","):
        if "-" in part:
            start, end = part.split("-")
            end = int(end) if end != "*" else total_chapters
            chapter_numbers.extend(range(int(start), end + 1))
        elif part != "*":
            chapter_numbers.append(int(part))
        else:
            return range(1, total_chapters + 1)
    return chapter_numbers


# Assuming get_chapter_audio_for_chapter is defined elsewhere
# You will need to update it to accept voice and model as parameters


def gpt_prompt_for_chapter_analysis(chunk, last_chapter_title):
    """
    Analyzes a text chunk to identify chapters using GPT-4, with a fallback to GPT-3.5 if necessary.
    Returns the last identified chapter if no new chapters are found, along with the text provided in the response.

    :param chunk: Text chunk to be analyzed.
    :param last_chapter_title: Title of the last identified chapter to continue from.
    :return: A list of chapters found in the chunk, or the last chapter if no new chapters are found.
    """

    from openai import (
        BadRequestError,
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
        RateLimitError,
        InternalServerError,
        APIConnectionError,
        APITimeoutError,
    )

    # Example JSON structure showing potential multiple chapters
    example_json = {
        "chapters": [
            {
                "chapter_title": "Chapter 1",
                "chapter_content": "Full found Content of Chapter 1...",
            },
            {
                "chapter_title": "Chapter 2",
                "chapter_content": "Full found Content of Chapter 2...",
            },
        ]
    }

    # Detailed prompt construction for GPT models
    prompt = (
        f"You are an helpfull AI assistant. You are helping to find the structure of a book inside a text."
        f"You are given a chunk of text. This text needs to be analysed."
        f"A chunk can contain a a chapter title but does not need to start with it."
        f"If the text does not start with a new chapter title use this title ->'{last_chapter_title}'<- for the text until you find a new chapter. "
        f"Chapter Titles usually are written in CAPITAL LETTERS and formed as a question."
        f"They also usually take a whole line."
        f"Be carful not to include any other text in the chapter title and also that in the text the chapter titles are somethimes mentioned. DO NOT include those mentions in the chapter title."
        f"Examine the text for any new chapter, and return their titles and full content. It is absolutly crucial that you return the full content of the chapters."
        f"No not change any of the text simply copy and past it."
        f"Be carfull not to add any styling to the text like /n or /t"
        f"Here is the text chunk for analysis: {chunk}."
        f"Again If no new chapters are found, simply use this ->'{last_chapter_title}'<- for the rest of the found chapter content. "
        f"Your response should be in a JSON format similar to this example: {json.dumps(example_json)}"
        f"You can do this. Give this your best shot. Take time to think. "
    )

    client = openai.OpenAI()  # Ensure the OpenAI client is set up with an API key

    attempts = 0
    max_attempts = 2
    models = ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]

    while attempts < max_attempts + 1:
        model = models[attempts % len(models)]
        # print(f"Sending the following detailed prompt to {model}:")
        # print(prompt)

        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Please respond with a detailed analysis in JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        response_content = response.choices[0].message.content
        # print(f"Received response from {model}:")
        # print(response_content)

        try:
            response_data = json.loads(response_content)

            return response_data  # Correct response with new chapters

        except BadRequestError:
            print("Bad request to OpenAI. Please check the request format.")
        except AuthenticationError:
            print("Authentication failed. Please check your OpenAI API key.")
        except PermissionDeniedError:
            print("Permission denied. Please check your access rights.")
        except NotFoundError:
            print("Requested resource not found.")
        except RateLimitError:
            print("Rate limit exceeded. Please try again later.")
        except (InternalServerError, APIConnectionError, APITimeoutError) as e:
            print(f"A server or connection error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        attempts += 1

    print("Failed to get a valid response after multiple attempts.")
    return []  # Return an empty list only if all attempts fail


def split_into_chunks(text, chunk_size=400):
    """
    Splits the book text into manageable chunks, trying to break at sentence endings.
    'chunk_size' is in characters, adjust based on testing.
    """
    chunks = []
    chunk_count = 0
    while text:
        # Take the first 'chunk_size' characters from the book text
        chunk = text[:chunk_size]

        # Ensure the chunk ends on a complete sentence where possible
        last_end = max(chunk.rfind("."), chunk.rfind("!"), chunk.rfind("?"))
        if last_end != -1 and len(chunk) - last_end < 200:
            # Adjust chunk to end at the last complete sentence
            chunk = chunk[: last_end + 1]
            # Adjust the remaining book text starting after the last complete sentence
            text = text[last_end + 1 :]
        else:
            # If no sentence ending is found, or it's too close to the end of the chunk, proceed as usual
            text = text[chunk_size:]

        chunks.append(chunk)
        chunk_count += 1

        # Print each chunk with spacing
        # print(f"Chunk {chunk_count}:\n{chunk}\n\n---\n")

    return chunks


def combine_chapter_responses(response_list):
    """
    Combines the chapter information from multiple responses into one list.
    If the same chapter appears in multiple responses, their content is combined.
    Assumes each response in response_list is already a list of dictionaries.
    """
    chapter_dict = {}
    for response in response_list:
        if isinstance(response, list):
            for chapter in response:
                title = chapter.get("chapter_title", "Untitled")
                content = chapter.get("chapter_content", "")

                if title in chapter_dict:
                    # Append content to existing chapter
                    # print(f"Appending content to existing chapter: {title}")
                    chapter_dict[title] += content
                else:
                    # Add new chapter
                    # print(f"Adding new chapter: {title}")
                    chapter_dict[title] = content
        else:
            print("Unexpected response format. Expected a list of dictionaries.")

    # Convert the dictionary back to a list of chapter dictionaries
    combined_chapters = [
        {"chapter_title": title, "chapter_content": content}
        for title, content in chapter_dict.items()
    ]
    print("Finished combining chapters.")
    return combined_chapters


import re


def word_list(text):
    # Split text into words, considering punctuation as separate entities
    return re.findall(r"\b\w+\b|\S", text.lower())


def get_chapters_for_text(text, book_title="Untitled"):
    print("Processing entire book...")

    chunks = split_into_chunks(text)
    all_chapters = []
    last_chapter_title = ""  # Initialize with an empty string

    for chunk_index, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_index + 1}: {chunk}")
        response = gpt_prompt_for_chapter_analysis(chunk, last_chapter_title)
        chapters = response.get("chapters", [])

        combined_chapter_words = []

        for chapter in chapters:
            print(f"Found chapter: {chapter.get('chapter_title')}")
            print(f"Chapter content: {chapter.get('chapter_content')}")

            title = chapter.get("chapter_title", "Untitled")
            content = chapter.get("chapter_content", "")
            last_chapter_title = title
            combined_chapter_words.extend(word_list(title + " " + content))

            chapter_found = False
            for chapter_dict in all_chapters:
                if title == chapter_dict.get("chapter_title"):
                    chapter_found = True
                    chapter_dict["chapter_content"] += " " + content
                    break
            if not chapter_found:
                all_chapters.append(
                    {"chapter_title": title, "chapter_content": content}
                )

        chunk_words = word_list(chunk)
        missing_words = [
            word for word in chunk_words if word not in combined_chapter_words
        ]

        if missing_words:
            print(f"Missing words in chunk {chunk_index + 1}: {missing_words}")

    return all_chapters


def split_into_subchapters(chapter_content, max_length=4000):
    """
    Splits a long chapter into subchapters based on a maximum character length.
    Tries to split at paragraph ends for natural breaks.
    """
    subchapters = []
    current_subchapter = ""

    for paragraph in chapter_content.split("\n"):
        if len(current_subchapter) + len(paragraph) + 1 > max_length:
            subchapters.append(current_subchapter)
            current_subchapter = paragraph
        else:
            current_subchapter += "\n" + paragraph

    # Add the last subchapter if it contains text
    if current_subchapter.strip():
        subchapters.append(current_subchapter)

    return subchapters


# Function to sanitize filenames
def sanitize_filename(filename):
    """Remove or replace invalid characters for file names."""
    invalid_chars = r'[<>:"/\\|?*]'  # Regex pattern for invalid filename characters
    return re.sub(
        invalid_chars, "_", filename
    )  # Replace invalid characters with underscore


def sanitize_filename_from_title(title, chapter_number, subchapter_number=1):
    sanitized_title = re.sub(r'[<>:"/\\|?*]', "_", title)
    filename = f"{chapter_number:02d}_{sanitized_title}"
    if subchapter_number > 1:
        filename += f"_{subchapter_number:02d}"
    return filename


import tkinter as tk
from tkinter import ttk, scrolledtext
import threading


def create_gui():
    """
    Initializes and displays the main GUI window.
    """
    root = tk.Tk()
    root.title("Text to Speech Converter")

    # Setup the main GUI components
    setup_main_gui(root)

    root.mainloop()


def clean_text(filedata, strings_to_remove):
    """
    General cleaning of the text.

    This function can be expanded with more specific cleaning requirements, such as removing
    repeating words or specific non-book related text. Additional logic or regex patterns can be
    implemented as needed.

    Args:
    filedata (str): The text to be cleaned.
    strings_to_remove (list of str): A list of strings to remove from the text.

    Returns:
    str: The cleaned text.
    """

    filedata = remove_page_numbers(filedata)

    filedata = remove_specific_strings(filedata, strings_to_remove)
    # Add more cleaning logic here if needed
    return filedata


def remove_specific_strings(text, strings_to_remove):
    """
    Remove specific strings from the text.

    This function iterates over a list of strings and removes each one from the text. This is useful
    for removing specific words or phrases that are known and defined in advance.

    Args:
    text (str): The original text from which strings will be removed.
    strings_to_remove (list of str): A list of strings that should be removed from the text.

    Returns:
    str: The text with specified strings removed.
    """
    for string in strings_to_remove:
        text = text.replace(string, "")
    return text


def remove_page_numbers(text):
    """
    Remove page numbers from the text.

    This function uses a regular expression to identify and remove patterns that match page numbers.
    The pattern '- Seite X von 471 -' is targeted, where X can be any number. This pattern is based on
    the example provided and can be modified to fit different page number formats.

    Args:
    text (str): The text from which page numbers will be removed.

    Returns:
    str: The text with page numbers removed.
    """
    pattern = r"- Seite \d+ von 471 -"
    return re.sub(pattern, "", text)


def flatten_bgb_structure(bgb_structure):
    chapters = []
    for book in bgb_structure:
        book_title = book["title"]
        for section in book["sections"]:
            section_title = section["title"]
            for title in section["titles"]:
                title_title = title["title"]
                for paragraph in title["paragraphs"]:
                    paragraph_title = paragraph["title"]
                    paragraph_content = paragraph["content"]
                    chapter_title = (
                        f"{book_title}_{section_title}_{title_title}_{paragraph_title}"
                    )
                    chapters.append(
                        {
                            "chapter_title": chapter_title,
                            "chapter_content": paragraph_content,
                        }
                    )
    return chapters


def split_bgb_text(text):
    # Reguläre Ausdrücke für die verschiedenen Komponenten
    book_regex = r"\n(Buch \d+[\s\S]*?)(?=\nBuch \d+|$)"
    section_regex = r"\n(Abschnitt \d+[\s\S]*?)(?=\nAbschnitt \d+|$)"
    title_regex = r"\n(Titel|Untertitel) \d+[\s\S]*?(?=(Titel|Untertitel) \d+|$)"
    paragraph_regex = r"\n§\s\d+\s[^§]*"

    bgb_structure = []

    # Alle Bücher finden
    books = re.findall(book_regex, text, re.MULTILINE)
    for book_content in books:
        book_split = book_content.strip().split("\n", 1)
        book_title = book_split[0] if len(book_split) > 1 else "Buch ohne Titel"
        book_content = book_split[1] if len(book_split) > 1 else ""

        book_dict = {"title": book_title, "sections": []}
        sections = re.findall(section_regex, book_content, re.MULTILINE)
        for section_content in sections:
            section_split = section_content.strip().split("\n", 1)
            section_title = (
                section_split[0] if len(section_split) > 1 else "Abschnitt ohne Titel"
            )
            section_content = section_split[1] if len(section_split) > 1 else ""

            section_dict = {"title": section_title, "titles": []}
            titles = re.findall(title_regex, section_content, re.MULTILINE)
            for title_content in titles:
                title_split = title_content.strip().split("\n", 1)
                title_title = (
                    title_split[0] if len(title_split) > 1 else "Titel ohne Titel"
                )
                title_content = title_split[1] if len(title_split) > 1 else ""

                title_dict = {"title": title_title, "paragraphs": []}
                paragraphs = re.findall(paragraph_regex, title_content, re.MULTILINE)
                for paragraph_content in paragraphs:
                    paragraph_split = paragraph_content.strip().split("\n", 1)
                    paragraph_title = (
                        paragraph_split[0]
                        if len(paragraph_split) > 1
                        else "Paragraph ohne Titel"
                    )
                    paragraph_content = (
                        paragraph_split[1] if len(paragraph_split) > 1 else ""
                    )

                    paragraph_dict = {
                        "title": paragraph_title,
                        "content": paragraph_content,
                    }
                    title_dict["paragraphs"].append(paragraph_dict)

                section_dict["titles"].append(title_dict)

            book_dict["sections"].append(section_dict)

        bgb_structure.append(book_dict)

    return bgb_structure


def extract_chapters_from_text(text):
    chapters = []
    lines = text.split("\n")
    chapter_counter = 1
    current_title = ""
    current_content = []

    for line in lines:
        # Check for Buch, Abschnitt, Titel, and start a new chapter
        if re.match(r"(Buch \d+|Abschnitt \d+|Titel \d+)", line):
            # Save previous chapter if it exists
            if current_title:
                chapters.append(
                    {
                        "unique_id": f"Chapter{chapter_counter}",
                        "chapter_title": current_title,
                        "chapter_content": " ".join(current_content).strip(),
                    }
                )
                chapter_counter += 1

            current_title = line.strip()
            current_content = []

        # Check for '§' and start a new chapter
        elif re.match(r"§ \d+", line):
            # Save previous chapter if it exists
            if current_title:
                chapters.append(
                    {
                        "unique_id": f"Chapter{chapter_counter}",
                        "chapter_title": current_title,
                        "chapter_content": " ".join(current_content).strip(),
                    }
                )
                chapter_counter += 1

            current_title = line.strip()
            current_content = []

        else:
            current_content.append(line.strip())

    # Add the last chapter
    if current_title:
        chapters.append(
            {
                "unique_id": f"Chapter{chapter_counter}",
                "chapter_title": current_title,
                "chapter_content": " ".join(current_content).strip(),
            }
        )

    return chapters


def setup_main_gui(root):
    """
    Sets up the main GUI components including mode selection and text input area.

    :param root: The root window of the tkinter application.
    """
    root.grid_columnconfigure(0, weight=1)  # Make the main column expandable

    # Mode selection
    mode_label = tk.Label(root, text="Select Mode:")
    mode_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

    mode_combobox = ttk.Combobox(root, values=["Normal", "Book", "Clean"])
    mode_combobox.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

    # Book title entry (initially hidden)
    book_title_label = tk.Label(root, text="Book Title:")
    book_title_entry = tk.Entry(root)

    # Function to show/hide book title entry based on mode
    def on_mode_change(event):
        mode = mode_combobox.get()
        if mode == "Book":
            book_title_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)
            book_title_entry.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        elif mode == "Clean":
            with open("BGB.txt", "r", encoding="utf-8") as file:
                filedata = file.read()

                # List of specific strings to remove
                strings_to_remove = [
                    "Ein Service des Bundesministeriums der Justiz sowie des Bundesamts für Justiz ‒ www.gesetze-im-internet.de",
                    # Add more unwanted phrases as needed
                ]

                filedata = clean_text(filedata, strings_to_remove)

                # bgb_structure = split_bgb_text(filedata)

                # chapters = flatten_bgb_structure(bgb_structure)
                chapters = extract_chapters_from_text(filedata)

                print(f"Found {len(chapters)} chapters.")

                # first 10 chapters

                # chapters = chapters[:10]
                """                 
                for chapter in chapters[:40]:
                print(chapter) 
                """

                book_title = "BGB"
                voice_model_matrix = get_default_voice_model_matrix("*")

                # Process each chapter
                chapter_audios = get_chapters_audio_for_chapters(
                    chapters, book_title, voice_model_matrix
                )

                # Call the check_audio_files_existence function
                check_audio_files_existence(chapters, book_title, voice_model_matrix)

        else:
            book_title_label.grid_remove()
            book_title_entry.grid_remove()

    mode_combobox.bind("<<ComboboxSelected>>", on_mode_change)

    # Text area for input
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD)
    text_area.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

    # Button row configuration
    button_frame = tk.Frame(root)
    button_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=1)

    # Start button for initiating conversion
    start_button = tk.Button(
        button_frame,
        text="Start",
        command=lambda: start_conversion_wrapper(
            mode_combobox, text_area, book_title_entry, root
        ),
    )
    start_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    # Load Book button
    load_book_button = tk.Button(
        button_frame, text="Load Book", command=lambda: load_book(root)
    )
    load_book_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    def open_empty_review_gui():
        empty_chapters = []  # Empty list of chapters
        empty_book_title = ""  # Empty book title
        display_chapters_for_review(empty_chapters, empty_book_title, root)

    # New Book button
    new_book_button = tk.Button(
        button_frame, text="New Book", command=open_empty_review_gui
    )
    new_book_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")


def load_book(root):
    global global_book_title  # Reference the global variable

    books_folder = "books"
    os.makedirs(books_folder, exist_ok=True)  # Ensure the books folder exists

    # Open a dialog to select a book file
    book_file = filedialog.askopenfilename(
        initialdir=books_folder,
        title="Select Book",
        filetypes=(("JSON Files", "*.json"), ("All Files", "*.*")),
    )

    if book_file:
        # Load the selected book
        with open(book_file, "r", encoding="utf-8") as file:
            chapters = json.load(file)

        # Update the global book title
        global_book_title = os.path.splitext(os.path.basename(book_file))[0].replace(
            "_", " "
        )

        display_chapters_for_review(chapters, global_book_title, root)


def start_conversion_wrapper(mode_combobox, text_area, book_title_entry, root):
    mode = mode_combobox.get()
    input_text = text_area.get("1.0", tk.END).strip()
    book_title = book_title_entry.get().strip() if mode == "Book" else ""

    def process_text():
        chapters = get_chapters_for_text(input_text, book_title)  # Pass book title
        display_chapters_for_review(chapters, book_title, root)  # Pass book title

    threading.Thread(target=process_text).start()

    # Function to save chapters to a JSON file
    import os
    import tkinter.filedialog as filedialog


def save_chapters_to_json(book_title, chapters):
    try:
        books_folder = "books"
        os.makedirs(books_folder, exist_ok=True)

        json_filename = (
            f"{book_title.replace(' ', '_')}.json" if book_title else "chapters.json"
        )
        json_filepath = os.path.join(books_folder, json_filename)

        with open(json_filepath, "w", encoding="utf-8") as file:
            json.dump(chapters, file, indent=4)

        print(f"Chapters saved to {json_filepath}")
    except Exception as e:
        print(f"Error saving chapters: {e}")


def display_chapters_for_review(chapters, book_title, root):
    review_window = tk.Toplevel(root)
    review_window.title("Review Chapters")
    current_chapter_index = 0

    # Layout configuration for resizing
    review_window.grid_columnconfigure(1, weight=1)
    review_window.grid_rowconfigure(1, weight=1)

    # Chapter list for navigation (made larger)
    chapter_list = tk.Listbox(review_window, width=40)  # Adjust width as needed
    chapter_list.grid(row=0, column=0, rowspan=4, sticky="nsew", padx=5, pady=5)
    for chapter in chapters:
        chapter_list.insert(tk.END, chapter.get("chapter_title", "Untitled"))

    # Function to update the display of the current chapter
    def update_chapter_display(index):
        chapter = chapters[index]
        chapter_title_var.set(chapter.get("chapter_title", "Untitled"))
        chapter_text_area.delete("1.0", tk.END)
        chapter_text_area.insert(tk.END, chapter.get("chapter_content", ""))

    # Function to update chapter titles in the list
    def refresh_chapter_list():
        chapter_list.delete(0, tk.END)
        for chapter in chapters:
            chapter_list.insert(tk.END, chapter.get("chapter_title", "Untitled"))

    # Update chapter data when the text or title is modified
    def update_chapter_data():
        current_chapter = chapters[current_chapter_index]
        current_chapter["chapter_title"] = chapter_title_var.get()
        current_chapter["chapter_content"] = chapter_text_area.get(
            "1.0", tk.END
        ).strip()
        refresh_chapter_list()  # Refresh the list to show updated titles

    # Function to handle chapter list selection
    def on_chapter_select(event):
        nonlocal current_chapter_index
        selection = chapter_list.curselection()
        if selection:
            current_chapter_index = selection[0]
            update_chapter_display(current_chapter_index)

    # Function to add a new chapter
    def add_new_chapter():
        new_chapter = {"chapter_title": "New Chapter", "chapter_content": ""}
        chapters.append(new_chapter)
        refresh_chapter_list()
        chapter_list.selection_set(len(chapters) - 1)  # Select the new chapter
        update_chapter_display(len(chapters) - 1)  # Display the new chapter

    # Function to delete the current chapter
    def delete_current_chapter():
        nonlocal current_chapter_index
        if 0 <= current_chapter_index < len(chapters):
            del chapters[current_chapter_index]
            refresh_chapter_list()
            new_index = min(current_chapter_index, len(chapters) - 1)
            if new_index >= 0:
                chapter_list.selection_set(new_index)
                update_chapter_display(new_index)
            else:
                chapter_title_var.set("")
                chapter_text_area.delete("1.0", tk.END)

    chapter_list.bind("<<ListboxSelect>>", on_chapter_select)

    # Editable chapter title
    chapter_title_var = tk.StringVar()
    chapter_title_entry = tk.Entry(review_window, textvariable=chapter_title_var)
    chapter_title_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

    # Chapter text area
    chapter_text_area = scrolledtext.ScrolledText(
        review_window, wrap=tk.WORD, height=5, width=50
    )
    chapter_text_area.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

    # Navigation buttons
    previous_button = tk.Button(
        review_window, text="Previous Chapter", command=lambda: change_chapter(-1)
    )
    previous_button.grid(row=2, column=1, sticky="w", padx=5, pady=5)

    next_button = tk.Button(
        review_window, text="Next Chapter", command=lambda: change_chapter(1)
    )
    next_button.grid(row=2, column=1, sticky="e", padx=5, pady=5)

    # Audio conversion buttons
    convert_current_button = tk.Button(
        review_window,
        text="Convert Current Chapter",
        command=lambda: convert_current_chapter(current_chapter_index),
    )
    convert_current_button.grid(row=3, column=1, sticky="w", padx=5, pady=5)

    convert_all_button = tk.Button(
        review_window,
        text="Convert All Chapters",
        command=lambda: convert_all_chapters(chapters),
    )
    convert_all_button.grid(row=3, column=1, sticky="e", padx=5, pady=5)

    def change_chapter(delta):
        nonlocal current_chapter_index
        new_index = current_chapter_index + delta
        if 0 <= new_index < len(chapters):
            current_chapter_index = new_index
            update_chapter_display(current_chapter_index)
            chapter_list.selection_clear(0, tk.END)
            chapter_list.selection_set(current_chapter_index)

    def convert_current_chapter(index):
        chapter = chapters[index]

        # Prompt user for chapter number
        chapter_number = simpledialog.askinteger(
            "Chapter Number", "Enter the chapter number:", parent=review_window
        )

        # Check if the user provided a chapter number
        if chapter_number is not None:
            # Proceed with audio conversion
            get_chapter_audio_for_chapter(chapter, chapter_number)

            # Mark the chapter as converted (e.g., change background color in the list)
            chapter_list.itemconfig(index, {"bg": "green"})
        else:
            # Handle case where user cancels the input or enters an invalid number
            print("Chapter conversion canceled or invalid chapter number entered.")

    def convert_all_chapters(chapters):
        # Implement conversion logic for all chapters
        start_audio_conversion(chapters)

    # Update chapter data when the text is modified
    def update_chapter_data():
        current_chapter = chapters[current_chapter_index]
        current_chapter["chapter_title"] = chapter_title_var.get()
        current_chapter["chapter_content"] = chapter_text_area.get(
            "1.0", tk.END
        ).strip()
        refresh_chapter_list()

    chapter_text_area.bind("<KeyRelease>", lambda event: update_chapter_data())
    chapter_title_entry.bind("<KeyRelease>", lambda event: update_chapter_data())

    # Add and delete chapter buttons
    add_chapter_button = tk.Button(
        review_window, text="Add Chapter", command=add_new_chapter
    )
    add_chapter_button.grid(row=5, column=1, sticky="w", padx=5, pady=5)

    delete_chapter_button = tk.Button(
        review_window, text="Delete Chapter", command=delete_current_chapter
    )
    delete_chapter_button.grid(row=5, column=1, sticky="e", padx=5, pady=5)

    # Button to save chapters to JSON
    save_json_button = tk.Button(
        review_window,
        text="Save Chapters to JSON",
        command=lambda: save_chapters_to_json(book_title, chapters),
    )
    save_json_button.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

    update_chapter_display(current_chapter_index)


def start_audio_conversion(chapters):
    """
    Starts the audio conversion process for the reviewed chapters.

    :param chapters: List of reviewed chapters.
    """
    get_chapters_audio_for_chapters(chapters, global_book_title)


def ask_for_api_key(root):
    """
    Asks the user for the OpenAI API key and saves it to a .env file.

    :param root: The root window of the tkinter application.
    :return: The entered API key.
    """
    api_key = simpledialog.askstring(
        "API Key Required", "Enter your OpenAI API key:", parent=root
    )

    # Save the key to a .env file
    with open(".env", "w") as file:
        file.write(f"OPENAI_API_KEY={api_key}\n")

    return api_key


def load_api_key():
    """
    Loads the OpenAI API key from the environment or prompts the user to enter it.

    :return: The OpenAI API key.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        # Initialize a minimal Tkinter root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Ask the user for the API key
        api_key = ask_for_api_key(root)

        # Destroy the Tkinter root window
        root.destroy()

    return api_key


if __name__ == "__main__":
    api_key = load_api_key()

    # Check if the user provided the key
    if not api_key:
        print("No API key provided. Exiting.")
        exit(1)

    # Initialize OpenAI client with your API key
    openai.api_key = api_key

    create_gui()
