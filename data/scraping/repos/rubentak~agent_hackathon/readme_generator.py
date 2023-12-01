'''
This LangChain Agent searches csv tables and replies with summarizations and aggregations
'''

#%% ---------------------------------------------  IMPORTS  ----------------------------------------------------------#
import streamlit as st
import openai
from credentials import OPENAI_API_KEY
from main import rec_streamlit, speak_answer, get_transcript_whisper
import time
import en_core_web_sm
import spacy_streamlit
import pathlib
import glob
import pandas as pd
import os
import tempfile
from langchain.agents import create_csv_agent
from langchain.agents import create_vectorstore_agent
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from pathlib import Path

#%% ----------------------------------------  LANGCHAIN PRELOADS -----------------------------------------------------#
embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=OPENAI_API_KEY)
llm_csv = OpenAI(openai_api_key=OPENAI_API_KEY)


# --------------------  SETTINGS  -------------------- #
st.set_page_config(page_title="Home", layout="wide")
st.markdown("""<style>.reportview-container .main .block-container {max-width: 95%;}</style>""", unsafe_allow_html=True)


# --------------------- HOME PAGE -------------------- #
st.title("README GENERATOR 📚📖")
st.write("""Use the power of LLMs with LangChain and OpenAI to scan through your documents. Find information 
and insight's with lightning speed. 🚀 Create new content with the support of state of the art language models and 
and voice command your way through your documents. 🎙️
This LangChain Agent searches a directory for markdown files and generates a README.md file with the most important content.
""")
st.write("Let's start interacting with GPT-4!")

temp_slider = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key="temperature")
recording_slider = st.sidebar.slider("Recording time", min_value=4, max_value=10, value=6, step=1, key="recording_time")


# ----------------- SIDE BAR SETTINGS ---------------- #
st.sidebar.subheader("Settings:")
tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=False)
ner_enabled = st.sidebar.checkbox("Enable NER in Response", value=False)

def run_GPT4(systemprompt, prompt, temperature):
    """Run GPT4 with the prompt and return the response"""
    openai.api_key = OPENAI_API_KEY
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=temperature,
        messages=[
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": prompt},
        ]
    )
    answer = completion.choices[0].message.content
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    total_tokens = completion.usage.total_tokens
    model = completion.model
    transcript_model = 'whisper-1'
    return answer, prompt_tokens, completion_tokens, total_tokens, model, transcript_model


#%% -----------------------------------------  DIRECTORY SCAN  -------------------------------------------------------#
def list_files_in_directory(path):
    with st.spinner('Scanning the directory...'):
        path = pathlib.Path(path)
        filepaths = [str(p) for p in path.rglob('*')]
        df = pd.DataFrame(filepaths, columns=['Filepath'])
        df['Filepath'] = df['Filepath'].apply(lambda x: pathlib.Path(x))
        df['Filename'] = df['Filepath'].apply(lambda x: x.name)
        df['Fileextension'] = df['Filepath'].apply(lambda x: x.suffix)
        df['Filesize'] = df['Filepath'].apply(lambda x: x.stat().st_size)
        df['Filecreationdate'] = df['Filepath'].apply(lambda x: x.stat().st_ctime)
        df['Filemodificationdate'] = df['Filepath'].apply(lambda x: x.stat().st_mtime)
        df['Filecreationdate'] = pd.to_datetime(df['Filecreationdate'], unit='s')
        df['Filemodificationdate'] = pd.to_datetime(df['Filemodificationdate'], unit='s')
        df['Modifiedthisweek'] = df['Filemodificationdate'].apply(lambda x: 'This week!' if x > pd.Timestamp.now() - pd.Timedelta(days=7) else 'Older')
        df['Filecreationdate'] = df['Filecreationdate'].dt.date
        df['Filemodificationdate'] = df['Filemodificationdate'].dt.date
        df['Fileorfolder'] = df['Fileextension'].apply(lambda x: 'Folder' if x == '' else 'File')
        df['Hidden'] = df['Filename'].apply(lambda x: 'Hidden' if x.startswith('.') else 'Not hidden')
        df['Parentfolder'] = df['Filepath'].apply(lambda x: x.parent.name)
        df['Filepath'] = df['Filepath'].apply(lambda x: str(x))
    return df


#%% -----------------------------------------  DIRECTORY VIZ  --------------------------------------------------------#
class DisplayablePath(object):
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None):
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        # Specify criteria to exclude directories/files
        children = sorted(list(path
                               for path in root.glob('*')  # for path in root.iterdir() <- all files
                               if not path.name.startswith('.') and criteria(path)),  # if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))


# Call the function with the path to the directory you want to print and
def tree_with_exception(path, criteria=None):
    if criteria is None:
        criteria = lambda path: not (path.name.startswith('.') or path.name == '__pycache__' or path.name == 'venv')
    paths = DisplayablePath.make_tree(Path.home() / path, criteria=criteria)
    for path in paths:
        # Save the output to a text string with linebreaks to use later
        tree = '\n'.join([path.displayable() for path in paths])
    return tree



# ------------------- FILE HANDLER ------------------- #
folder_path = st.text_input('Enter the folder path:')
if folder_path:
    if os.path.isdir(folder_path):
        files_df = list_files_in_directory(folder_path)
        st.write(f'Found {len(files_df)} files in the directory:')
        with st.expander("Document Expander (Press button on the right to fold or unfold)", expanded=True):
            st.dataframe(files_df)

        # Save the DataFrame to a CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            csv_file_path = f.name
            files_df.to_csv(csv_file_path, index=False)

    else:
        st.error('Invalid folder path. Please enter a valid path.')


show_structure = st.button("Show folder structure")
if show_structure:
    structure = tree_with_exception(folder_path)
    with st.expander("Document Expander (Press button on the right to fold or unfold)", expanded=True):
        st.code(structure)

# --------------------- USER INPUT --------------------- #
user_input = st.text_area("Describe the Project:")
# If record button is pressed, rec_streamlit records and the output is saved
audio_bytes = rec_streamlit()

systemprompt = f'Write an interesting README.md with the header points:\n# 🧭 Project Overview \n## 🚧 Prerequisites\n## 🎛 Project Setup\n## 📦 Project Structure\n## 🗄️ Data\n## 📚 References\n## 🏆 Conclusion\n## 🤝 Contributions\n\nAnd the content summary:'

# ------------------- TRANSCRIPTION -------------------- #
# if audio_bytes or user_input:

if audio_bytes or user_input:

    audio_used = False
    if audio_bytes:
        audio_used = True
        try:
            with open("audio.wav", "wb") as file:
                file.write(audio_bytes)
        except Exception as e:
            st.write("Error recording audio:", e)
        transcript = get_transcript_whisper("audio.wav")
    else:
        transcript = user_input

    st.write("**Recognized:**")
    st.write(transcript)

    if any(word in transcript for word in ["abort recording"]):
        st.write("... Script stopped by user")
        exit()
    # ----------------------- ANSWER ----------------------- #
    with st.spinner("Fetching answer ..."):
        time.sleep(6)


    answer = run_GPT4(systemprompt, transcript, temp_slider)
    # Extrct answer from GPT4 output tuple
    answer = answer[0]

    st.markdown(answer)
    structure = tree_with_exception(folder_path)
    st.code(structure)
    speak_answer(answer, tts_enabled)
    st.success("**Interaction finished**")

    save_readme = st.button("Save README.md")

    # If save button is pressed, save the README.md file + the structure in a README.md in the folder
    if save_readme:
        # Create a README.md file
        with open(folder_path + "/README.md", "w") as file:
            file.write(answer)
            file.write(structure)
        st.success("README.md saved in the folder")


    #delete audio_bytes
    audio_bytes = None

    # delete user input
    user_input = None


