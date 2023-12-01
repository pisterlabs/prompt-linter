import os  # For IO
import sys  # For IO
import shutil  # For IO
import subprocess  # For IO
import openai  # The good stuff
import time  # For logging
from time import sleep  # Zzz
import toml  # For parsing settings
import json  # For parsing JSON
import pyperclip  # For copying to clipboard
import re  # For regex
import requests  # For accessing the web
from bs4 import BeautifulSoup  # For parsing HTML
import readline
import argparse  # For parsing command line arguments

# Speech recognition
import threading
import speech_recognition
import sounddevice

# Voice output
from google.cloud import texttospeech

# Document loaders
from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

# Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector stores
from langchain.vectorstores import Chroma

# Embeddings
from langchain.embeddings import OpenAIEmbeddings

# Memory
from langchain.memory import ConversationBufferMemory

# Chat models
from langchain.chat_models import ChatOpenAI

# LLM
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback


# Formatting
from rich.console import (
    Console,
    OverflowMethod,
)  # https://rich.readthedocs.io/en/stable/console.html
from rich.theme import Theme
from rich.spinner import Spinner
from rich import print
from rich.padding import Padding
from rich.table import Table
from rich import box
from rich.syntax import Syntax


# {{{ Logging
class Logger:
    LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}

    def __init__(self, filename):
        self.filename = filename
        self.level = self.LOG_LEVELS["INFO"]
        try:
            open(filename, "a").close()
            self.file = open(filename, "a")
            self.file.truncate(0)
        except Exception as e:
            print(f"Error opening log file: {e}")

    def log(self, message: str, level: str = "INFO"):
        if level is not None:
            level_value = self.LOG_LEVELS[level]
        else:
            level_value = self.level

        if level_value >= self.level:
            write = self.file.write(f"{time.ctime()} [{level}] {message}\n")
            if write:
                self.file.flush()
            else:
                print(f"Error writing to log file: {write}")

    def close(self):
        self.file.close()


log = Logger("./neuma.log")
# }}}


# {{{ ChatModel
class ChatModel:
    def __init__(self):
        self.config = self.get_config()
        self.mode = "normal"  # Default mode
        self.persona = self.set_persona("default")
        self.voice_output = False  # Default voice output
        self.voice = self.config["voices"]["english"]  # Default voice
        self.vector_db = self.config["vector_db"]["default"]  # Default vector db

    # {{{ Get config
    def get_config(self) -> dict:
        """Get config from config.toml and API keys from .env"""
        # Get config
        if os.path.isfile(os.path.expanduser("~/.config/neuma/config.toml")):
            config_path = os.path.expanduser("~/.config/neuma/config.toml")
        else:
            config_path = os.path.dirname(os.path.realpath(__file__)) + "/config.toml"
        try:
            with open(config_path, "r") as f:
                config = toml.load(f)
        except Exception as e:
            log.log("Error loading config: {}".format(e))
            return e

        # Get API keys
        if os.path.isfile(os.path.expanduser("~/.config/neuma/.env")):
            env_path = os.path.expanduser("~/.config/neuma/.env")
        else:
            env_path = os.path.dirname(os.path.realpath(__file__)) + "/.env"
        try:
            with open(env_path, "r") as f:
                env = toml.load(f)
                # OpenAI
                openai_api_key = env["OPENAI_API_KEY"]
                if openai_api_key:
                    config["openai"]["api_key"] = openai_api_key
                    log.log("OpenAI API key loaded : {}".format(config["openai"]))
                # Google app
                google_app_api_key = env["GOOGLE_APPLICATION_CREDENTIALS"]
                if google_app_api_key:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_app_api_key
                else:
                    raise ValueError("No API key found.")
        except Exception as e:
            log.log("Error loading API keys: {}".format(e))
            return e

        return config

    # }}}

    # {{{ Generate final prompt
    def generate_final_message(self, user_prompt: str) -> list:
        """Generate final prompt (messages) for OpenAI API"""

        # Add a dot at the end of the prompt if there isn't one
        if user_prompt[-1] not in ["?", "!", "."]:
            user_prompt += "."
        self.user_prompt = user_prompt

        # Conversation up to this point
        conversation = self.conversation
        # log.log("conversation: {}".format(conversation))

        # {{{ Persona identity
        if not conversation:
            log.log("Persona : {}".format(self.persona))
            persona_identity = self.get_persona_identity()
            for message in persona_identity:
                conversation.append(message)
        # }}}

        # {{{ Mode instructions
        log.log("Mode : {}".format(self.mode))
        mode_instructions = self.config["modes"][self.mode]
        if mode_instructions:
            # Replace # with all the text after # in the user_prompt
            hashtag = self.find_hashtag(self.user_prompt)
            log.log("hashtag: {}".format(hashtag))
            if hashtag:
                mode_instructions = mode_instructions.replace("#", hashtag)
            mode_instructions_message = {"role": "system", "content": mode_instructions}
            conversation.append(mode_instructions_message)
            log.log("Mode instructions : {}".format(mode_instructions_message))
        # }}}

        # {{{ File content to insert
        if "~{f:" in user_prompt and "}~" in user_prompt:
            file_path = user_prompt.split("~{f:")[1].split("}~")[0]
            log.log("file_path: {}".format(file_path))
            if os.path.isfile(file_path):
                with open(file_path, "r") as f:
                    file_content = f.read()
                    user_prompt = user_prompt.replace(
                        "~{f:" + file_path + "}~", file_content
                    )
                    log.log("user_prompt: {}".format(user_prompt))
            else:
                log.log("File not found")
        # }}}

        # {{{ URL content to insert
        if "~{w:" in user_prompt and "}~" in user_prompt:
            url = user_prompt.split("~{w:")[1].split("}~")[0]
            log.log("url: {}".format(url))
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    log.log("response: {}".format(response))
                    soup = BeautifulSoup(response.text, "html.parser")
                    url_content = soup.get_text()
                    # remove all newlines and extra spaces
                    url_content = re.sub(r"\s+", " ", url_content)
                    if len(url_content) > 3000:
                        url_content = url_content[:3000]
                    user_prompt = user_prompt.replace("~{w:" + url + "}~", url_content)
                    log.log("user_prompt: {}".format(user_prompt))
                else:
                    log.log("Error getting URL content")
            except Exception as e:
                log.log("Error getting URL content: {}".format(e))
        # }}}

        # {{{ User input
        user_prompt = {"role": "user", "content": user_prompt}
        conversation.append(user_prompt)
        log.log("User prompt : {}".format(user_prompt))
        log.log("Final messages: {}".format(conversation))
        # }}}

        return conversation

    # }}}

    # {{{ Generate response from OpenAI API
    def generate_response(self, messages: list) -> str | Exception:
        """Generate response from OpenAI API"""

        prompt = json.dumps(messages)

        api_key = self.config["openai"]["api_key"]
        # log.log("api_key: {}".format(api_key))

        model = self.config["openai"]["model"]
        # log.log("model: {}".format(model))

        temperature = self.config["openai"]["temperature"]
        # log.log("temperature: {}".format(temperature))

        top_p = self.config["openai"]["top_p"]
        # log.log("top_p: {}".format(top_p))

        max_tokens = self.config["openai"]["max_tokens"]
        # log.log("max_tokens: {}".format(max_tokens))

        persist_folder = self.config["vector_db"]["persist_folder"]

        vector_db_name = self.vector_db
        full_path = os.path.join(persist_folder, vector_db_name)

        # Log messages :
        # log.log("messages: {}".format(messages))

        # If variable "chat_history" does not exist, create it
        if "chat_history" not in globals():
            chat_history = []

        # LLM
        try:
            llm = ChatOpenAI(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                # top_p=top_p,
            )
            # log.log("llm: {}".format(llm))
        except Exception as e:
            return e

        # Embeddings
        try:
            embeddings = OpenAIEmbeddings(
                openai_api_key=os.environ["OPENAI_API_KEY"],
            )
            # log.log("embeddings: {}".format(embeddings))
        except Exception as e:
            return e

        # Vector store
        try:
            full_path = os.path.join(persist_folder, vector_db_name)
            vector_db = Chroma(
                persist_directory=full_path,
                embedding_function=embeddings,
            )
            # log.log("vector_db: {}".format(vector_db))
        except Exception as e:
            return e

        # Chain
        try:
            retriever = vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2},
            )
            llm = ChatOpenAI(model_name=model)

            qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

            with get_openai_callback() as callback:
                response = qa({"question": prompt, "chat_history": chat_history})
                # response = chain({"question": prompt, "chat_history": chat_history})
                response_data = {
                    "id": "",
                    "created": "",
                    "status": "success",
                    "message": response["answer"],
                    "promptTokens": callback.prompt_tokens,
                    "completionTokens": callback.completion_tokens,
                    "totalTokens": callback.total_tokens,
                    # 'sourceDocuments': response['source_documents'][0],
                }
                log.log("response_data: {}".format(response_data))
        except Exception as e:
            return e

        # Add to conversation
        response_message = {"role": "assistant", "content": response_data["message"]}
        self.conversation.append(response_message)

        chat_history.append(response_data["message"])

        # Process the response
        self.processed_response = self.process_response(response_data["message"])

        return self.processed_response

    # }}}

    # {{{ Process response
    def process_response(self, response: str) -> str:
        """Process response, formats the response"""

        # {{{ General formating

        # Remove double line breaks
        response = response.replace("\n\n", "\n")

        # Keep only what is between ``` and ```
        if "```" in response:
            response = response.split("```")[1]
            response = response.split("```")[0]

        # }}}

        # {{{ Table mode formatting
        if self.mode == "table":
            # Remove everything before the first |
            if "|" in response:
                response = response.split("|", 1)[1]
                response = "|" + response

            log.log("response: {}".format(response))

            lines = response.split("\n")
            lines = list(filter(None, lines))

            # remove lines that contain '---'
            lines = [line for line in lines if "---" not in line]

            # Create table
            table = Table(show_lines=True)

            # Add columns
            for column in lines[0].split("|"):
                table.add_column(column)

            # Add rows
            for row in lines[1:]:
                cells = row.split("|")
                table.add_row(*cells)

            # Return table
            return table

        # }}}

        # {{{ Code mode formatting
        elif self.mode == "code":
            language = self.find_hashtag(self.user_prompt)
            syntax = Syntax(
                response,
                language,
                line_numbers=False,
                theme="gruvbox-dark",
                word_wrap=True,
            )
            return syntax
        # }}}

        # {{{ CSV mode formatting
        elif self.mode == "csv":
            separator = self.find_hashtag(self.user_prompt)
            response = response.replace(",", separator)

        return response
        # }}}

    # }}}

    # {{{ Personae

    # List personae
    def list_personae(self) -> str | dict | Exception:
        """List the available personae from the personae file"""
        if os.path.isfile(os.path.expanduser("~/.config/neuma/personae.toml")):
            personae_path = os.path.expanduser("~/.config/neuma/personae.toml")
            log.log("Personae path : {}".format(personae_path))
        else:
            personae_path = (
                os.path.dirname(os.path.realpath(__file__)) + "/personae.toml"
            )
            log.log("Personae path : {}".format(personae_path))
        try:
            with open(personae_path, "r") as f:
                personae = toml.load(f)
                log.log("Personae : {}".format(personae))
        except Exception as e:
            return e
        return personae

    # Set persona
    def set_persona(self, persona: str) -> bool | Exception:
        # TODO: Check if persona exists before setting
        self.persona = persona
        temperature = self.get_persona_temperature()
        self.set_temperature(temperature)
        return True

    # Get persona
    def get_persona(self) -> str:
        return self.persona

    # Get persona prompt
    def get_persona_identity(self) -> list:
        if self.persona != "":
            personae = self.list_personae()
            for persona in personae["persona"]:
                if persona["name"] == self.persona:
                    persona_identity = persona["messages"]
                    log.log("Persona identity : {}".format(persona_identity))
        else:
            persona_identity = ""
        return persona_identityidentity

    # Get persona temperature
    def get_persona_temperature(self) -> float:
        if self.persona != "":
            personae = self.list_personae()
            for persona in personae["persona"]:
                if persona["name"] == self.persona:
                    temperature = persona["temp"]
        else:
            temperature = ""
        return temperature

    # OBS: Get persona language code
    # def get_persona_language_code(self) -> str:
    #     if self.persona != "":
    #         personae = self.list_personae()
    #         for persona in personae["persona"]:
    #             if persona["name"] == self.persona:
    #                 language_code = persona["language_code"]
    #     else:
    #         language_code = ""
    #     return language_code

    # OBS: Get persona voice name
    # def get_persona_voice_name(self) -> str:
    #     if self.persona != "":
    #         personae = self.list_personae()
    #         for persona in personae["persona"]:
    #             if persona["name"] == self.persona:
    #                 voice_name = persona["voice_name"]
    #     else:
    #         voice_name = ""
    #     return voice_name

    # }}}

    # {{{ Conversation

    # Create new conversation
    def new_conversation(self) -> list:
        self.conversation = []

    # Save conversation, write it to a file
    def save_conversation(self, filename: str) -> bool:
        data_folder = self.config["conversations"]["data_folder"]
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        try:
            with open(data_folder + filename + ".neu", "w") as f:
                output = ""
                for message in self.conversation:
                    output += message["content"] + "\n\n"
                f.write(output)
        except Exception as e:
            return e
        return True

    # List conversations
    def list_conversations(self) -> list:
        data_folder = self.config["conversations"]["data_folder"]
        try:
            files = [f for f in os.listdir(data_folder) if f.endswith(".neu")]
        except Exception as e:
            return e
        return files

    # Open conversation
    def open_conversation(self, filename: str) -> str:
        data_folder = self.config["conversations"]["data_folder"]
        # Get list of files
        try:
            with open(data_folder + filename + ".neu", "r") as f:
                self.conversation = f.read()

        except Exception as e:
            return e
        return True

    # Trash conversation
    def trash_conversation(self, filename: str) -> bool:
        data_folder = self.config["conversations"]["data_folder"]
        # Get list of files
        try:
            os.remove(data_folder + filename + ".neu")
        except Exception as e:
            return e
        return True

    # }}}

    # {{{ Modes

    # Get mode
    def get_mode(self) -> str:
        return self.mode

    # Set mode
    def set_mode(self, mode: str) -> None:
        # Check if mode exists before setting
        modes = self.list_modes()
        if mode in modes:
            self.mode = mode
        else:
            raise ValueError("No mode with that name found.")

    # List modes
    def list_modes(self) -> list:
        return self.config["modes"].keys()

    # Find hashtag in prompt
    def find_hashtag(self, prompt: str) -> str | bool:
        words = prompt.split(" ")
        for word in words:
            if word.startswith("#"):
                log.log("Hashtag found : {}".format(word))
                return word[1:]
        return False

    # }}}

    # {{{ Models

    # {{{ List models
    def list_models(self) -> list:
        models = openai.Model.list()
        return models

    # }}}

    # }}}

    # {{{ Voice input
    def listen(self) -> str:
        self.config = self.get_config()
        self.input_device = self.config["audio"]["input_device"]
        self.input_timeout = self.config["audio"]["input_timeout"]
        log.log("input_timeout: {}".format(self.input_timeout))
        self.input_limit = self.config["audio"]["input_limit"]
        log.log("input_limit: {}".format(self.input_limit))

        log.log("Listening...")

        # https://github.com/Uberi/speech_recognition
        recognizer = speech_recognition.Recognizer()

        log.log("input_device: {}".format(self.input_device))
        with speech_recognition.Microphone(device_index=self.input_device) as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(
                    source,
                    timeout=self.input_timeout,
                    phrase_time_limit=self.input_limit,
                )

                with open("tmp.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                audio_file = open("tmp.wav", "rb")

                # Transcribe audio to text https://platform.openai.com/docs/guides/speech-to-text
                transcription = self.transcribe(audio_file)

                return transcription

            except speech_recognition.WaitTimeoutError as e:
                return e

    # }}}

    # {{{ Transcribe
    def transcribe(self, audio_file: str) -> str:
        api_key = self.config["openai"]["api_key"]
        model = "whisper-1"
        prompt = ""
        response_format = "json"
        temperature = 0
        language = self.voice[:2]
        try:
            transcription = openai.Audio.transcribe(
                api_key=api_key,
                model=model,
                file=audio_file,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                language=language,
            )
            transcription = transcription["text"]
            return transcription

        except speech_recognition.RequestError as e:
            return e

    # }}}

    # {{{ Voice output
    def get_voice_output(self) -> str:
        return self.voice_output

    def set_voice_output(self, voice_output: str) -> None:
        self.voice_output = voice_output

    def get_voices(self) -> list:
        self.voices = self.config["voices"]
        return self.voices

    def get_voice(self) -> str:
        return self.voice

    def set_voice(self, voice: str) -> None:
        self.voice = self.config["voices"][voice]

    def speak(self, response: str) -> None:
        if self.voice_output == True:
            # Instantiates a client
            client = texttospeech.TextToSpeechClient()

            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=response)

            # if a persona is set
            # if self.persona != "":
            #     voice_name = self.get_persona_voice_name()
            # else:
            voice_name = self.get_voice()

            language_code = voice_name[:5]

            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                # https://cloud.google.com/text-to-speech/docs/voices
                language_code=language_code,
                name=voice_name,
            )

            # Select the type of audio file you want returned
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            # Perform the text-to-speech request
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            # The response's audio_content is binary.
            with open("tmp.mp3", "wb") as out:
                # Write the response to the output file.
                out.write(response.audio_content)

                # play with mpv without output
                subprocess.run(["mpv", "--really-quiet", "tmp.mp3"])

                # remove the audio file
                os.remove("tmp.mp3")

    # }}}

    # {{{ Document loader
    def load_document(self, file_path: str) -> list | Exception:
        """Load document from file"""

        # Loader mapping
        LOADER_MAPPING = {
            ".csv": (CSVLoader, {}),
            ".doc": (UnstructuredWordDocumentLoader, {}),
            ".docx": (UnstructuredWordDocumentLoader, {}),
            ".epub": (UnstructuredEPubLoader, {}),
            ".html": (UnstructuredHTMLLoader, {}),
            ".md": (UnstructuredMarkdownLoader, {}),
            ".odt": (UnstructuredODTLoader, {}),
            ".pdf": (PyMuPDFLoader, {}),
            ".ppt": (UnstructuredPowerPointLoader, {}),
            ".pptx": (UnstructuredPowerPointLoader, {}),
            ".txt": (TextLoader, {"encoding": "utf8"}),
        }

        # Get file extension
        _, file_extension = os.path.splitext(file_path)

        # Check if file extension is supported
        if file_extension not in LOADER_MAPPING:
            return Exception("File extension not supported.")

        # Load loader corresponding to file extension
        Loader, kwargs = LOADER_MAPPING[file_extension]
        try:
            loader = Loader(file_path, **kwargs)
        except Exception as e:
            return e

        # Load document
        try:
            document = loader.load()
        except Exception as e:
            return e

        return document

    # }}}

    # {{{ Text splitter
    def split_text(self, document: list) -> list | Exception:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=10
            )
            documents = text_splitter.split_documents(document)

        except Exception as e:
            return e

        return documents

    # }}}

    # {{{ Embed
    def embed_doc(self, documents: list) -> list | Exception:
        embeddings_model = self.config["embeddings"]["model"]
        try:
            embeddings = OpenAIEmbeddings(
                openai_api_key=os.environ["OPENAI_API_KEY"], model=embeddings_model
            )

        except Exception as e:
            return e

        return embeddings

    # }}}

    # {{{ Store as vector
    def store_as_vector(
        self, documents: list, embeddings: OpenAIEmbeddings
    ) -> list | Exception:
        persist_folder = self.config["vector_db"]["persist_folder"]
        if not os.path.exists(persist_folder):
            log.log("Vector store folder doesn't exist, creating it")
            os.makedirs(persist_folder)

        vector_db_name = self.vector_db
        full_path = os.path.join(persist_folder, vector_db_name)
        log.log("full_path: {}".format(full_path))

        try:
            # if vector store exists
            if os.path.exists(full_path):
                log.log("Vector store exists")
                try:
                    vector_db = Chroma(
                        persist_directory=full_path,
                        embedding_function=embeddings,
                    )
                    log.log("Vector store loaded")
                except Exception as e:
                    return e

                try:
                    vector_db.add_documents(documents)
                    log.log("Document added to vector store")
                except Exception as e:
                    return e

            # if vector store doesn't exist, create it
            else:
                log.log("Vector store doesn't exist")
                vector_db = Chroma.from_documents(
                    documents,
                    embeddings,
                    persist_directory=full_path,
                )
            collection = vector_db._collection.get()
            doc_ids = collection["ids"]
        except Exception as e:
            return e

        # Save vector store to disk
        try:
            vector_db.persist()
            log.log("Vector store saved to disk")
        except Exception as e:
            return e

        return doc_ids

    # }}}

    # {{{ Get vector dbs
    def get_vector_dbs(self) -> list:
        persist_folder = self.config["vector_db"]["persist_folder"]
        return os.listdir(persist_folder)

    # }}}

    # {{{ Get vector db
    def get_vector_db(self) -> str:
        return self.vector_db

    # }}}

    # {{{ Set vector db
    def set_vector_db(self, vector_db: str) -> None:
        self.vector_db = vector_db
        full_path = os.path.join(
            self.config["vector_db"]["persist_folder"], self.vector_db
        )
        if not os.path.exists(full_path):
            os.mkdir(full_path)

    # }}}

    # {{{ Trash vector db
    def trash_vector_db(self, vector_db: str) -> bool | Exception:
        persist_folder = self.config["vector_db"]["persist_folder"]
        try:
            shutil.rmtree(persist_folder + "/" + vector_db)
        except Exception as e:
            return e
        return True

    # }}}

    # {{{ Copy to clipboard
    def copy_to_clipboard(self, selection) -> None:
        output = ""
        if isinstance(selection, list):
            for message in selection:
                output += message["content"] + "\n\n"
        else:
            output = selection
        pyperclip.copy(output)

    # }}}

    # {{{ Set temperature
    def set_temperature(self, temperature: float) -> bool:
        self.config["openai"]["temperature"] = float(temperature)
        return True

    # }}}

    # {{{ Set top_p
    def set_top(self, top: float) -> bool:
        self.config["openai"]["top_p"] = float(top)
        return True

    # }}}

    # {{{ Set max_tokens
    def set_max_tokens(self, max_tokens: int) -> bool:
        self.config["openai"]["max_tokens"] = int(max_tokens)
        return True

    # }}}


# }}}


# {{{ ChatView
class ChatView:
    def __init__(self):
        self.chat_controller = None
        self.config = None
        self.console = None
        self.chat_controller = None

    def display_message(self, message: str, style: str) -> None:
        """Display message"""
        output = Padding(message, (0, 2))
        self.console.print(output, style=style)

    def clear_screen(self) -> None:
        """Clear screen"""
        os.system("cls" if os.name == "nt" else "clear")

    def display_help(self) -> None:
        """Display help table with list of commands and aliases"""
        help_table = Table(box=box.SQUARE)
        help_table.add_column("Command", max_width=20)
        help_table.add_column("Description")
        help_table.add_row("h", "Display this help section")
        help_table.add_row("r", "Restart application")
        help_table.add_row("c", "List saved conversations")
        help_table.add_row("c \[conversation]", "Open conversation \[conversation]")
        help_table.add_row("cc", "Create a new conversation")
        help_table.add_row(
            "cs [conversation]", "Save the current conversation as [conversation]"
        )
        help_table.add_row("ct \[conversation]", "Trash conversation \[conversation]")
        help_table.add_row("cy", "Copy current conversation to clipboard")
        help_table.add_row("m", "List available modes")
        help_table.add_row("m \[mode]", "Switch to mode \[mode]")
        help_table.add_row("p", "List available personae")
        help_table.add_row("p \[persona]", "Switch to persona \[persona]")
        help_table.add_row("l", "List available languages")
        help_table.add_row("l \[language]", "Set language to \[language]")
        help_table.add_row("vi", "Switch to voice input")
        help_table.add_row("vo", "Switch on voice output")
        help_table.add_row("e \[document]", "Embed \[path/to/document]")
        help_table.add_row("d", "List available vector dbs")
        help_table.add_row("d \[db]", "Create or switch to vector db \[db]")
        help_table.add_row("dt \[db]", "Trash vector db \[db]")
        help_table.add_row("y", "Copy last answer to clipboard")
        help_table.add_row("t", "Get the current temperature value")
        help_table.add_row("t \[temp]", "Set the temperature to \[temp]")
        help_table.add_row("tp", "Get the current top_p value")
        help_table.add_row("tp \[top_p]", "Set the top_p to \[top_p]")
        help_table.add_row("mt", "Get the current max_tokens value")
        help_table.add_row("mt \[max_tokens]", "Set the max_tokens to \[max_tokens]")
        help_table.add_row("lm", "List available microphones")
        help_table.add_row("cls", "Clear the screen")
        help_table.add_row("q", "Quit")
        self.console.print(help_table)

    def display_response(self, response: str) -> None:
        """Display response in chat view or speak it"""

        # Display the response
        self.display_message(response, "answer")

        # Speak the response
        self.chat_controller.speak(response)

        # New line
        self.console.print()


# }}}


# {{{ ChatController
class ChatController:
    def __init__(self, chat_model, chat_view):
        self.chat_model = chat_model
        self.chat_view = chat_view

        self.chat_view.config = self.chat_model.config

        self.chat_view.chat_controller = self

        self.input_mode = "text"

        self.console = Console(
            theme=Theme(self.chat_model.config["theme"]),
            record=True,
            color_system="truecolor",
        )
        self.chat_view.console = self.console

    # {{{ Startup
    def start(self):
        """Start the chat"""
        # Clear the screen
        self.chat_view.clear_screen()

        # Parse command line arguments
        if len(sys.argv) > 1:
            self.parse_command_line_arguments(sys.argv[1:])

        # Create a new conversation
        self.chat_model.new_conversation()

        # Parse command
        while True:
            user_input = self.chat_view.console.input("> ")
            self.parse_command(user_input)

    # }}}

    # {{{ Parse command line arguments
    def parse_command_line_arguments(self, arguments: list) -> None:
        """Parse command line arguments"""

        # get config
        self.config = self.chat_model.config

        # Create an ArgumentParser object
        parser = argparse.ArgumentParser(
            description="neuma is a minimalistic ChatGPT interface for the command line."
        )

        # Define arguments
        parser.add_argument("-i", "--input", help="Input prompt")
        parser.add_argument("-p", "--personae", help="Set personae")
        parser.add_argument("-m", "--mode", help="Set mode")
        parser.add_argument("-t", "--temp", help="Set temperature")

        # Parse the command line arguments
        args = parser.parse_args()

        # Set personae
        if args.personae:
            self.chat_model.set_persona(args.personae)

        # Set mode
        if args.mode:
            self.chat_model.set_mode(args.mode)

        # Set temperature
        if args.temp:
            self.chat_model.set_temperature(args.temp)

        # Prompt input
        if args.input:
            self.chat_model.new_conversation()
            final_message = self.chat_model.generate_final_message(args.input)
            response = self.chat_model.generate_response(final_message)
            print(response)
            exit()

    # }}}

    # {{{ Parse command
    def parse_command(self, command: str) -> None:
        """Parse the user input and execute the command"""

        # {{{ System

        # {{{ Exit
        if command == "q":
            self.exit_app()
        # }}}

        # {{{ Restart
        elif command == "r":
            self.chat_view.display_message("Restarting...", "success")
            time.sleep(1)
            python = sys.executable
            os.execl(python, python, *sys.argv)
        # }}}

        # {{{ Help
        elif command == "h":
            self.chat_view.display_help()
        # }}}

        # {{{ Clear screen
        elif command == "cls":
            self.chat_view.clear_screen()
        # }}}

        # {{{ Copy answer to clipboard
        elif command == "y":
            # log conversation
            if len(self.chat_model.conversation) > 0:
                last_message = self.chat_model.conversation[-1].get("content")
                log.log("Last message: {}".format(last_message))
                self.chat_model.copy_to_clipboard(last_message)
                self.chat_view.display_message(
                    "Copied last answer to clipboard.", "success"
                )
            else:
                self.chat_view.display_message("Nothing to copy to clipboard.", "error")
        # }}}

        # {{{ Get temperature
        elif command == "t":
            self.chat_view.display_message(
                "Temperature: {}".format(
                    self.chat_model.config["openai"]["temperature"]
                ),
                "info",
            )
        # }}}

        # {{{ Set temperature
        elif command.startswith("t "):
            temp = command[2:]
            set_temp = self.chat_model.set_temperature(temp)
            if isinstance(set_temp, Exception):
                self.chat_view.display_message(
                    "Error setting temperature: {}".format(set_temp), "error"
                )
            else:
                self.chat_view.display_message(
                    "temperature set to {}.".format(temp), "success"
                )
        # }}}

        # {{{ Get top_p
        elif command == "tp":
            self.chat_view.display_message(
                "top_p: {}".format(self.chat_model.config["openai"]["top_p"]), "info"
            )
        # }}}

        # {{{ Set top_p
        elif command.startswith("tp "):
            top = command[2:]
            set_top = self.chat_model.set_top(top)
            if isinstance(set_top, Exception):
                self.chat_view.display_message(
                    "Error setting top_p: {}".format(set_top), "error"
                )
            else:
                self.chat_view.display_message(
                    "top_p set to {}.".format(top), "success"
                )
        # }}}

        # {{{ Get max tokens
        elif command == "mt":
            self.chat_view.display_message(
                "max_tokens: {}".format(self.chat_model.config["openai"]["max_tokens"]),
                "info",
            )
        # }}}

        # {{{ Set max tokens
        elif command.startswith("mt "):
            max_tokens = command[2:]
            set_max_tokens = self.chat_model.set_max_tokens(max_tokens)
            if isinstance(set_max_tokens, Exception):
                self.chat_view.display_message(
                    "Error setting max tokens: {}".format(set_max_tokens), "error"
                )
            else:
                self.chat_view.display_message(
                    "max_tokens set to {}.".format(max_tokens), "success"
                )
        # }}}

        # }}}

        # {{{ Conversations

        # {{{ List conversations
        elif command == "c":
            conversations_list = self.chat_model.list_conversations()
            if isinstance(conversations_list, Exception):
                self.chat_view.display_message(
                    "Error listing conversation: {}".format(conversations_list), "error"
                )
            else:
                # if there is at least one conversation
                if len(conversations_list) > 0:
                    self.chat_view.display_message("Conversations", "section")
                    for conversation in conversations_list:
                        self.chat_view.display_message(conversation, "info")
        # }}}

        # {{{ Create conversation
        elif command == "cc":
            self.chat_model.new_conversation()
            self.chat_model.set_persona("")
            self.chat_view.mode = "normal"
            self.chat_view.display_message("New conversation.", "success")
            time.sleep(1)
            self.chat_view.clear_screen()
        # }}}

        # {{{ Save conversation
        elif command.startswith("cs "):
            filename = command.split(" ")[-1]
            save = self.chat_model.save_conversation(filename)
            if isinstance(save, Exception):
                self.chat_view.display_message(
                    "Error saving conversation: {}".format(save), "error"
                )
            else:
                self.chat_view.display_message("Conversation saved.", "success")
        # }}}

        # {{{ Open conversation
        elif command.startswith("c "):
            filename = command.split(" ")[-1]
            if filename == "":
                self.chat_view.display_message("Please specify a filename.", "error")
            open_conversation = self.chat_model.open_conversation(filename)
            if isinstance(open_conversation, Exception):
                self.chat_view.display_message(
                    "Error opening conversation: {}".format(open_conversation), "error"
                )
            else:
                self.chat_view.display_message("Conversation opened.", "success")
                sleep(1)
                self.chat_view.clear_screen()
                self.chat_view.display_message(self.chat_model.conversation, "answer")
        # }}}

        # {{{ Trash conversation
        elif command.startswith("ct "):
            filename = command.split(" ")[-1]
            trash_conversation = self.chat_model.trash_conversation(filename)
            if isinstance(trash_conversation, Exception):
                self.chat_view.display_message(
                    "Error trashing conversation: {}".format(trash_conversation),
                    "error",
                )
            else:
                self.chat_view.display_message("Conversation trashed.", "success")
        # }}}

        # {{{ Copy conversation to clipboard
        elif command == "cy":
            self.chat_model.copy_to_clipboard(self.chat_model.conversation)
            self.chat_view.display_message(
                "Copied conversation to clipboard.", "success"
            )
        # }}}

        # }}}

        # {{{ Modes

        # {{{ List modes
        elif command == "m":
            modes = self.chat_model.list_modes()
            self.chat_view.display_message("Modes", "section")
            current_mode = self.chat_model.get_mode()
            for mode in modes:
                if mode == current_mode:
                    self.chat_view.display_message(mode + " <", "info")
                else:
                    self.chat_view.display_message(mode, "info")
        # }}}

        # {{{ Set mode
        elif command.startswith("m "):
            mode = command.split(" ")[1]
            try:
                set_mode = self.chat_model.set_mode(mode)
                self.chat_view.display_message(
                    "Mode set to {}.".format(mode), "success"
                )
            except Exception as e:
                self.chat_view.display_message(
                    "Error setting mode: {}".format(e), "error"
                )
        # }}}

        # }}}

        # {{{ Personae

        # {{{ List Personae
        elif command == "p":
            personae = self.chat_model.list_personae()
            self.chat_view.display_message("Personae", "section")
            current_persona = self.chat_model.get_persona()

            for persona in personae["persona"]:
                if persona["name"] == current_persona:
                    self.chat_view.display_message(persona["name"] + " <", "info")
                else:
                    self.chat_view.display_message(persona["name"], "info")
        # }}}

        # {{{ Set persona
        elif command.startswith("p "):
            persona = command.split(" ")[1]
            set_persona = self.chat_model.set_persona(persona)
            if isinstance(set_persona, Exception):
                self.chat_view.display_message(
                    "Error setting persona: {}".format(set_persona), "error"
                )
            else:
                self.chat_view.display_message(
                    "Persona set to {}.".format(persona), "success"
                )
                self.chat_model.new_conversation()
                sleep(1)
                self.chat_view.clear_screen()
        # }}}

        # }}}

        # {{{ Languages / Voice

        # {{{ Voice input
        elif command == "vi":
            # Toggle input mode
            if self.input_mode == "text":
                self.input_mode = "voice"
                self.chat_view.display_message("Voice input mode enabled.", "success")
                log.log("Voice input mode enabled.")

                # while in voice input mode
                while self.input_mode == "voice":
                    # Start spinner
                    with self.chat_view.console.status(""):
                        # Listen for voice input
                        self.voice_input = self.chat_model.listen()

                    # Stop spinner
                    self.chat_view.console.status("").stop()

                    if isinstance(self.voice_input, Exception):
                        self.chat_view.display_message(
                            "Error with voice input: {}".format(self.voice_input),
                            "error",
                        )
                    else:
                        # Display voice input
                        self.chat_view.display_message(self.voice_input, "prompt")

                        # if voice input is "Exit."
                        if self.voice_input == "Exit.":
                            self.input_mode = "text"
                            self.chat_model.set_voice_output(False)
                            self.chat_view.display_message(
                                "Voice input mode disabled.", "success"
                            )
                        else:
                            log.log("Processing voice input...")

                            # Start spinner
                            with self.chat_view.console.status(""):
                                # Generate final prompt
                                final_message = self.chat_model.generate_final_message(
                                    self.voice_input
                                )

                                # Generate response
                                response = self.chat_model.generate_response(
                                    final_message
                                )

                            # Stop spinner
                            self.chat_view.console.status("").stop()

                            # Display response
                            self.chat_view.display_response(response)

            else:
                self.input_mode = "text"
                self.chat_view.display_message("Voice input mode disabled.", "success")
        # }}}

        # {{{ Voice output
        elif command == "vo":
            self.chat_model.set_voice_output(not self.chat_model.get_voice_output())
            if self.chat_model.get_voice_output():
                self.chat_view.display_message("Voice output enabled.", "success")
            else:
                self.chat_view.display_message("Voice output disabled.", "success")
        # }}}

        # {{{ List languages / voices
        elif command == "l":
            voices = self.chat_model.get_voices()
            self.chat_view.display_message("Languages", "section")
            current_voice = self.chat_model.get_voice()
            for voice in voices:
                if self.chat_model.config["voices"][voice] == current_voice:
                    self.chat_view.display_message(voice + " <", "info")
                else:
                    self.chat_view.display_message(voice, "info")
        # }}}

        # {{{ Set language / voice
        elif command.startswith("l "):
            voice = command.split(" ")[1]
            set_voice = self.chat_model.set_voice(voice)
            if isinstance(set_voice, Exception):
                self.chat_view.display_message(
                    "Error setting language: {}".format(set_voice), "error"
                )
            else:
                self.chat_view.display_message(
                    "Language set to {}.".format(voice), "success"
                )
        # }}}

        # {{{ List microphones
        elif command.startswith("lm"):
            for index, name in enumerate(
                speech_recognition.Microphone.list_microphone_names()
            ):
                print(
                    'Microphone with name "{1}" found for `Microphone(device_index={0})`'.format(
                        index, name
                    )
                )
        # }}}

        # }}}

        # {{{ Documents

        # {{{ List vector dbs
        elif command == "d":
            vector_dbs = self.chat_model.get_vector_dbs()
            self.chat_view.display_message("Vector stores", "section")
            if len(vector_dbs) == 0:
                self.chat_view.display_message("No vector stores found.", "info")
            else:
                current_vector_db = self.chat_model.get_vector_db()
                for vector_db in vector_dbs:
                    if vector_db == current_vector_db:
                        self.chat_view.display_message(vector_db + " <", "info")
                    else:
                        self.chat_view.display_message(vector_db, "info")

        # }}}

        # {{{ Create or use vector db
        elif command.startswith("d "):
            vector_db = command.split(" ")[1]
            self.chat_model.set_vector_db(vector_db)
            self.chat_view.display_message(
                "Vector store set to {}.".format(vector_db), "success"
            )
        # }}}

        # {{{ Trash vector db
        elif command.startswith("dt "):
            vector_db = command.split(" ")[1]
            trash_vector_db = self.chat_model.trash_vector_db(vector_db)
            if isinstance(trash_vector_db, Exception):
                self.chat_view.display_message(
                    "Error trashing vector store: {}".format(trash_vector_db),
                    "error",
                )
            else:
                self.chat_view.display_message("Vector store trashed.", "success")
        # }}}

        # {{{ Embed document
        elif command.startswith("e "):
            file_path = command.split(" ")[1]

            # Load document
            try:
                document = self.chat_model.load_document(file_path)
                log.log("Loaded document: {}".format(file_path))
            except Exception as e:
                self.chat_view.display_message(
                    "Error loading document: {}".format(e), "error"
                )

            # Split text
            try:
                chunks = self.chat_model.split_text(document)
                log.log("Document split into {} chunks.".format(len(chunks)))
            except Exception as e:
                self.chat_view.display_message(
                    "Error splitting text: {}".format(e), "error"
                )

            # Embed document
            try:
                embeddings = self.chat_model.embed_doc(chunks)
                log.log("Document embedded")
            except Exception as e:
                self.chat_view.display_message(
                    "Error embedding document: {}".format(e), "error"
                )

            # Store as vector
            try:
                doc_ids = self.chat_model.store_as_vector(chunks, embeddings)
                log.log("Document stored as vector!")
            except Exception as e:
                self.chat_view.display_message(
                    "Error storing document as vector: {}".format(e), "error"
                )

            # Display message
            self.chat_view.display_message("Document embedded.", "success")

        # }}}

        # }}}

        # Normal prompt
        else:
            # Start spinner
            with self.chat_view.console.status(""):
                # Generate final prompt
                try:
                    final_message = self.chat_model.generate_final_message(command)

                    # Generate response
                    try:
                        response = self.chat_model.generate_response(final_message)

                        # Display response
                        self.chat_view.display_response(response)

                    # Error generating response
                    except Exception as e:
                        self.chat_view.display_message(
                            "Error generating response: {}".format(e), "error"
                        )

                # Error generating final prompt
                except Exception as e:
                    self.chat_view.display_message(
                        "Error generating final message: {}".format(e), "error"
                    )

    # }}}

    # {{{ Speak
    def speak(self, text):
        """Speak the text."""
        self.chat_model.speak(text)

    # }}}

    # Exit
    def exit_app(self):
        """Exit the app."""
        self.chat_view.display_message("Exiting neuma, goodbye!", "success")
        sleep(1)
        self.chat_view.console.clear()
        exit()


# }}}


# {{{ Main
def main():
    # clear console
    os.system("touch neuma.log")
    log.log("-------------- Starting neuma --------------")

    # Model
    chat_model = ChatModel()

    # View
    chat_view = ChatView()

    # Controller
    chat_controller = ChatController(chat_model, chat_view)

    # Start the controller
    chat_controller.start()


# if __name__ in { "__main__", "__mp_main__" }:
if __name__ == "__main__":
    main()
# }}}
