from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from pathlib import Path
import os

import characters
import settings_private

DEBUG_MODE = True

# Logging

LOG_FILE = "pippa.log"

# Project
PROJECT_TITLE = f"Pippa - Personalized, Ingesting, Persistent, Polymorphic, Adaptive GPT Chatbot"
VERSION = "1.1 - Final Public Release"

# ZEP Long-term Memory
ZEP_API_URL = settings_private.ZEP_API_URL

# Paths
BASE_DIR = Path(__file__).resolve().parent
CONVERSATION_SAVE_FOLDER = BASE_DIR / "conversations"
TEMP_FOLDER = BASE_DIR / "temp"
IMAGES_FOLDER = BASE_DIR / "images"
DOCUMENT_FOLDER = str(BASE_DIR / "docs")
CHROMA_DB_FOLDER = str(BASE_DIR / "chromadb")
USER_INPUT_SAVE_FILE = str(TEMP_FOLDER / "last_user_input.md")
SNAPSHOT_FILENAME = "snapshot.json"

# Folders to make if they don't exist

FOLDERS_TO_MAKE = [
    'static',
    str(CONVERSATION_SAVE_FOLDER),
    str(TEMP_FOLDER),
    DOCUMENT_FOLDER,
    CHROMA_DB_FOLDER,
]

# GPT Models
DEFAULT_GPT_MODEL = "gpt-4"
DEFAULT_GPT_MODEL_TEMPERATURE = 0.0
DEFAULT_GPT_HELPER_MODEL = "gpt-3.5-turbo-16k"
DEFAULT_GPT_QA_HELPER_MODEL = "gpt-3.5-turbo-16k"
DEFAULT_GPT_AGENT_HELPER_MODEL = "gpt-4"
DEFAULT_GPT_QA_HELPER_MODEL_TEMPERATURE = 0
DEFAULT_GPT_HELPER_MODEL_TEMPERATURE = 0.0
DEFAULT_GPT_AGENT_HELPER_MODEL_TEMPERATURE = 0.0
MIN_TOKENS = 100
DEFAULT_TOKENS = 500
DEFAULT_MIN_COMPLETION_TOKENS = 200
DEFAULT_COMPLETION_TOKENS = 1000

MAX_TOKENS_4K = 4096
MAX_TOKENS_8K = 8132
MAX_TOKENS_16K = 16384
MAX_TOKENS_32K = 32768
TOKEN_SAFE_MARGIN = 200
MAX_TOKEN_LIMIT_FOR_SUMMARY = 1000
AVERAGE_USER_INPUT_TOKENS = 500

LLM_VERBOSE = False
STREAMING_ENABLED = True

MAX_NUM_CHARS_FOR_CUSTOM_INSTRUCTIONS = 1500

# Costs - per 1K tokens

TOKENS_PER_K = 1000
MODEL_COST_GPT4_8K_INPUT = 0.03 / TOKENS_PER_K
MODEL_COST_GPT4_8K_OUTPUT = 0.06 / TOKENS_PER_K
MODEL_COST_GPT3_TURBO_4K_INPUT = 0.0015 / TOKENS_PER_K
MODEL_COST_GPT3_TURBO_4K_OUTPUT = 0.002 / TOKENS_PER_K
MODEL_COST_GPT3_TURBO_16K_INPUT = 0.003 / TOKENS_PER_K
MODEL_COST_GPT3_TURBO_16K_OUTPUT = 0.004 / TOKENS_PER_K

# Memory

DEFAULT_MEMORY_TYPE = settings_private.DEFAULT_MEMORY_TYPE
MEMORY_TYPES = ["Sliding Window", "Token", "Summary", "Summary Buffer", "Zep", "Buffer"]
SLIDING_CONTEXT_WINDOW = 10

# Avatars
AVATAR_HUMAN = str(IMAGES_FOLDER / "human.png")
AVATAR_AI = str(IMAGES_FOLDER / "ai.png")
AVATAR_SYSTEM = str(IMAGES_FOLDER / "system.png")

# TTS
VOICE_NAME_AI = characters.AI_NAME
VOICE_ID_AI = settings_private.VOICE_ID_AI
VOICE_FILE_AI = str(TEMP_FOLDER / "_ai.mp3")
VOICE_NAME_SYSTEM = characters.SYSTEM_NAME
VOICE_ID_SYSTEM = settings_private.VOICE_ID_SYSTEM
VOICE_FILE_SYSTEM = str(TEMP_FOLDER / "_system.mp3")
VOICE_ID_HUMAN = settings_private.VOICE_ID_HUMAN
VOICE_NAME_HUMAN = characters.HUMAN_NAME
VOICE_FILE_HUMAN = str(TEMP_FOLDER / "_human.mp3")
VOICES_URL = "https://api.elevenlabs.io/v1/voices"
VOICE_URL_SYSTEM = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID_SYSTEM}"
VOICE_URL_AI = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID_AI}"
VOICE_URL_HUMAN = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID_HUMAN}"

# Audio
TRANSCRIPTION_TEMP_AUDIO_FILE = str(TEMP_FOLDER / "_transcribed_audio.wav")
AUDIO_SERVER_URL = settings_private.AUDIO_SERVER_URL
OPENAI_AUDIO_MODEL = "whisper-1"
AUDIO_SERVER_PORT = 5000

# Prompts Keywords
PROMPT_KEYWORD_PREFIX_SYSTEM = "system:"
PROMPT_KEYWORD_PREFIX_CI = "ci:"
PROMPT_KEYWORD_PREFIX_QA = "qa:"
PROMPT_KEYWORD_PREFIX_GOOGLE = "google:"
PROMPT_KEYWORD_PREFIX_WIKI = "wiki:"
PROMPT_KEYWORD_PREFIX_MATH = "math:"
PROMPT_KEYWORD_PREFIX_MIDJOURNEY = "midjourney:"

NORMAL_PROMPT_KEYWORD_PREFIXES = [
    PROMPT_KEYWORD_PREFIX_SYSTEM,
    PROMPT_KEYWORD_PREFIX_CI,
    PROMPT_KEYWORD_PREFIX_QA,
]

AGENT_PROMPT_PREFIXES = [
    PROMPT_KEYWORD_PREFIX_GOOGLE,
    PROMPT_KEYWORD_PREFIX_WIKI,
    PROMPT_KEYWORD_PREFIX_MATH,
    PROMPT_KEYWORD_PREFIX_MIDJOURNEY,
]

ALL_PROMPT_PREFIXES = NORMAL_PROMPT_KEYWORD_PREFIXES + AGENT_PROMPT_PREFIXES

# Vector DB

INGEST_THREADS = os.cpu_count() or 8
VECTORDB_COLLECTION = "pippa_long_term_memory"
CONVERSATION_FILE_JSON_SCHEMA = "."  # jq schema for JSON conversation files

DOCUMENTS_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}
DOCUMENT_SPLITTER_CHUNK_SIZE = 1000
DOCUMENT_SPLITTER_CHUNK_OVERLAP = 200
PYTHON_SPLITTER_CHUNK_SIZE = 880
PYTHON_SPLITTER_CHUNK_OVERLAP = 200
NUM_SOURCES_TO_RETURN = 5

RETRIEVER_TEMPLATE = """
    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

SHOW_SOURCES = True

# Agents

MAX_AGENTS_ITERATIONS = 5
