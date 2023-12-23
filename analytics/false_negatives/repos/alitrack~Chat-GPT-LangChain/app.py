import io
import os
import ssl
from contextlib import closing
from typing import Optional, Tuple
import datetime

import boto3
import gradio as gr
import requests

# UNCOMMENT TO USE WHISPER
import warnings
import whisper

from langchain import ConversationChain, LLMChain

from langchain.agents import load_tools, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from threading import Lock

# Console to variable
from io import StringIO
import sys
import re

from openai.error import AuthenticationError, InvalidRequestError, RateLimitError

# Pertains to Express-inator functionality
from langchain.prompts import PromptTemplate

from polly_utils import PollyVoiceData, NEURAL_ENGINE
from azure_utils import AzureVoiceData

# Pertains to question answering functionality
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain

news_api_key = os.environ["NEWS_API_KEY"]
tmdb_bearer_token = os.environ["TMDB_BEARER_TOKEN"]

TOOLS_LIST = ['serpapi', 'wolfram-alpha', 'pal-math', 'pal-colored-objects']  #'google-search','news-api','tmdb-api','open-meteo-api'
TOOLS_DEFAULT_LIST = ['serpapi', 'pal-math']
BUG_FOUND_MSG = "Congratulations, you've found a bug in this application!"
# AUTH_ERR_MSG = "Please paste your OpenAI key from openai.com to use this application. It is not necessary to hit a button or key after pasting it."
AUTH_ERR_MSG = "Please paste your OpenAI key from openai.com to use this application. "
MAX_TOKENS = 512

LOOPING_TALKING_HEAD = "videos/Masahiro.mp4"
TALKING_HEAD_WIDTH = "192"
MAX_TALKING_HEAD_TEXT_LENGTH = 155

# Pertains to Express-inator functionality
NUM_WORDS_DEFAULT = 0
MAX_WORDS = 400
FORMALITY_DEFAULT = "N/A"
TEMPERATURE_DEFAULT = 0.5
EMOTION_DEFAULT = "N/A"
LANG_LEVEL_DEFAULT = "N/A"
TRANSLATE_TO_DEFAULT = "N/A"
LITERARY_STYLE_DEFAULT = "N/A"
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["original_words", "num_words", "formality", "emotions", "lang_level", "translate_to",
                     "literary_style"],
    template="Restate {num_words}{formality}{emotions}{lang_level}{translate_to}{literary_style}the following: \n{original_words}\n",
)

POLLY_VOICE_DATA = PollyVoiceData()
AZURE_VOICE_DATA = AzureVoiceData()

# Pertains to WHISPER functionality
WHISPER_DETECT_LANG = "Detect language"


# UNCOMMENT TO USE WHISPER
warnings.filterwarnings("ignore")
WHISPER_MODEL = whisper.load_model("tiny")
print("WHISPER_MODEL", WHISPER_MODEL)


# UNCOMMENT TO USE WHISPER
def transcribe(aud_inp, whisper_lang):
    if aud_inp is None:
        return ""
    aud = whisper.load_audio(aud_inp)
    aud = whisper.pad_or_trim(aud)
    mel = whisper.log_mel_spectrogram(aud).to(WHISPER_MODEL.device)
    _, probs = WHISPER_MODEL.detect_language(mel)
    options = whisper.DecodingOptions()
    if whisper_lang != WHISPER_DETECT_LANG:
        whisper_lang_code = POLLY_VOICE_DATA.get_whisper_lang_code(whisper_lang)
        options = whisper.DecodingOptions(language=whisper_lang_code)
    result = whisper.decode(WHISPER_MODEL, mel, options)
    print("result.text", result.text)
    result_text = ""
    if result and result.text:
        result_text = result.text
    return result_text


# Temporarily address Wolfram Alpha SSL certificate issue
ssl._create_default_https_context = ssl._create_unverified_context


# TEMPORARY FOR TESTING
def transcribe_dummy(aud_inp_tb, whisper_lang):
    if aud_inp_tb is None:
        return ""
    # aud = whisper.load_audio(aud_inp)
    # aud = whisper.pad_or_trim(aud)
    # mel = whisper.log_mel_spectrogram(aud).to(WHISPER_MODEL.device)
    # _, probs = WHISPER_MODEL.detect_language(mel)
    # options = whisper.DecodingOptions()
    # options = whisper.DecodingOptions(language="ja")
    # result = whisper.decode(WHISPER_MODEL, mel, options)
    result_text = "Whisper will detect language"
    if whisper_lang != WHISPER_DETECT_LANG:
        whisper_lang_code = POLLY_VOICE_DATA.get_whisper_lang_code(whisper_lang)
        result_text = f"Whisper will use lang code: {whisper_lang_code}"
    print("result_text", result_text)
    return aud_inp_tb


# Pertains to Express-inator functionality
def transform_text(desc, express_chain, num_words, formality,
                   anticipation_level, joy_level, trust_level,
                   fear_level, surprise_level, sadness_level, disgust_level, anger_level,
                   lang_level, translate_to, literary_style):
    num_words_prompt = ""
    if num_words and int(num_words) != 0:
        num_words_prompt = "using up to " + str(num_words) + " words, "

    # Change some arguments to lower case
    formality = formality.lower()
    anticipation_level = anticipation_level.lower()
    joy_level = joy_level.lower()
    trust_level = trust_level.lower()
    fear_level = fear_level.lower()
    surprise_level = surprise_level.lower()
    sadness_level = sadness_level.lower()
    disgust_level = disgust_level.lower()
    anger_level = anger_level.lower()

    formality_str = ""
    if formality != "n/a":
        formality_str = "in a " + formality + " manner, "

    # put all emotions into a list
    emotions = []
    if anticipation_level != "n/a":
        emotions.append(anticipation_level)
    if joy_level != "n/a":
        emotions.append(joy_level)
    if trust_level != "n/a":
        emotions.append(trust_level)
    if fear_level != "n/a":
        emotions.append(fear_level)
    if surprise_level != "n/a":
        emotions.append(surprise_level)
    if sadness_level != "n/a":
        emotions.append(sadness_level)
    if disgust_level != "n/a":
        emotions.append(disgust_level)
    if anger_level != "n/a":
        emotions.append(anger_level)

    emotions_str = ""
    if len(emotions) > 0:
        if len(emotions) == 1:
            emotions_str = "with emotion of " + emotions[0] + ", "
        else:
            emotions_str = "with emotions of " + ", ".join(emotions[:-1]) + " and " + emotions[-1] + ", "

    lang_level_str = ""
    if lang_level != LANG_LEVEL_DEFAULT:
        lang_level_str = "at a level that a person in " + lang_level + " can easily comprehend, " if translate_to == TRANSLATE_TO_DEFAULT else ""

    translate_to_str = ""
    if translate_to != TRANSLATE_TO_DEFAULT:
        translate_to_str = "translated to " + translate_to + (
            "" if lang_level == TRANSLATE_TO_DEFAULT else " at a level that a person in " + lang_level + " can easily comprehend") + ", "

    literary_style_str = ""
    if literary_style != LITERARY_STYLE_DEFAULT:
        if literary_style == "Prose":
            literary_style_str = "as prose, "
        if literary_style == "Story":
            literary_style_str = "as a story, "
        elif literary_style == "Summary":
            literary_style_str = "as a summary, "
        elif literary_style == "Outline":
            literary_style_str = "as an outline numbers and lower case letters, "
        elif literary_style == "Bullets":
            literary_style_str = "as bullet points using bullets, "
        elif literary_style == "Poetry":
            literary_style_str = "as a poem, "
        elif literary_style == "Haiku":
            literary_style_str = "as a haiku, "
        elif literary_style == "Limerick":
            literary_style_str = "as a limerick, "
        elif literary_style == "Rap":
            literary_style_str = "as a rap, "
        elif literary_style == "Joke":
            literary_style_str = "as a very funny joke with a setup and punchline, "
        elif literary_style == "Knock-knock":
            literary_style_str = "as a very funny knock-knock joke, "
        elif literary_style == "FAQ":
            literary_style_str = "as a FAQ with several questions and answers, "

    formatted_prompt = PROMPT_TEMPLATE.format(
        original_words=desc,
        num_words=num_words_prompt,
        formality=formality_str,
        emotions=emotions_str,
        lang_level=lang_level_str,
        translate_to=translate_to_str,
        literary_style=literary_style_str
    )

    trans_instr = num_words_prompt + formality_str + emotions_str + lang_level_str + translate_to_str + literary_style_str
    if express_chain and len(trans_instr.strip()) > 0:
        generated_text = express_chain.run(
            {'original_words': desc, 'num_words': num_words_prompt, 'formality': formality_str,
             'emotions': emotions_str, 'lang_level': lang_level_str, 'translate_to': translate_to_str,
             'literary_style': literary_style_str}).strip()
    else:
        print("Not transforming text")
        generated_text = desc

    # replace all newlines with <br> in generated_text
    generated_text = generated_text.replace("\n", "\n\n")

    prompt_plus_generated = "GPT prompt: " + formatted_prompt + "\n\n" + generated_text

    print("\n==== date/time: " + str(datetime.datetime.now() - datetime.timedelta(hours=5)) + " ====")
    print("prompt_plus_generated: " + prompt_plus_generated)

    return generated_text


def load_chain(tools_list, llm):
    chain = None
    express_chain = None
    memory = None
    if llm:
        print("\ntools_list", tools_list)
        tool_names = tools_list
        tools = load_tools(tool_names, llm=llm, news_api_key=news_api_key, tmdb_bearer_token=tmdb_bearer_token)

        memory = ConversationBufferMemory(memory_key="chat_history")

        chain = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=memory)
        express_chain = LLMChain(llm=llm, prompt=PROMPT_TEMPLATE, verbose=True)
    return chain, express_chain, memory


def set_openai_api_key(api_key):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key and api_key.startswith("sk-") and len(api_key) > 50:
        os.environ["OPENAI_API_KEY"] = api_key
        print("\n\n ++++++++++++++ Setting OpenAI API key ++++++++++++++ \n\n")
        print(str(datetime.datetime.now()) + ": Before OpenAI, OPENAI_API_KEY length: " + str(
            len(os.environ["OPENAI_API_KEY"])))
        llm = OpenAI(temperature=0, max_tokens=MAX_TOKENS)
        print(str(datetime.datetime.now()) + ": After OpenAI, OPENAI_API_KEY length: " + str(
            len(os.environ["OPENAI_API_KEY"])))
        chain, express_chain, memory = load_chain(TOOLS_DEFAULT_LIST, llm)

        # Pertains to question answering functionality
        embeddings = OpenAIEmbeddings()
        qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

        print(str(datetime.datetime.now()) + ": After load_chain, OPENAI_API_KEY length: " + str(
            len(os.environ["OPENAI_API_KEY"])))
        os.environ["OPENAI_API_KEY"] = ""
        return chain, express_chain, llm, embeddings, qa_chain, memory
    return None, None, None, None, None, None


def run_chain(chain, inp, capture_hidden_text):
    output = ""
    hidden_text = None
    if capture_hidden_text:
        error_msg = None
        tmp = sys.stdout
        hidden_text_io = StringIO()
        sys.stdout = hidden_text_io

        try:
            output = chain.run(input=inp)
        except AuthenticationError as ae:
            error_msg = AUTH_ERR_MSG + str(datetime.datetime.now()) + ". " + str(ae)
            print("error_msg", error_msg)
        except RateLimitError as rle:
            error_msg = "\n\nRateLimitError: " + str(rle)
        except ValueError as ve:
            error_msg = "\n\nValueError: " + str(ve)
        except InvalidRequestError as ire:
            error_msg = "\n\nInvalidRequestError: " + str(ire)
        except Exception as e:
            error_msg = "\n\n" + BUG_FOUND_MSG + ":\n\n" + str(e)

        sys.stdout = tmp
        hidden_text = hidden_text_io.getvalue()

        # remove escape characters from hidden_text
        hidden_text = re.sub(r'\x1b[^m]*m', '', hidden_text)

        # remove "Entering new AgentExecutor chain..." from hidden_text
        hidden_text = re.sub(r"Entering new AgentExecutor chain...\n", "", hidden_text)

        # remove "Finished chain." from hidden_text
        hidden_text = re.sub(r"Finished chain.", "", hidden_text)

        # Add newline after "Thought:" "Action:" "Observation:" "Input:" and "AI:"
        hidden_text = re.sub(r"Thought:", "\n\nThought:", hidden_text)
        hidden_text = re.sub(r"Action:", "\n\nAction:", hidden_text)
        hidden_text = re.sub(r"Observation:", "\n\nObservation:", hidden_text)
        hidden_text = re.sub(r"Input:", "\n\nInput:", hidden_text)
        hidden_text = re.sub(r"AI:", "\n\nAI:", hidden_text)

        if error_msg:
            hidden_text += error_msg

        print("hidden_text: ", hidden_text)
    else:
        try:
            output = chain.run(input=inp)
        except AuthenticationError as ae:
            output = AUTH_ERR_MSG + str(datetime.datetime.now()) + ". " + str(ae)
            print("output", output)
        except RateLimitError as rle:
            output = "\n\nRateLimitError: " + str(rle)
        except ValueError as ve:
            output = "\n\nValueError: " + str(ve)
        except InvalidRequestError as ire:
            output = "\n\nInvalidRequestError: " + str(ire)
        except Exception as e:
            output = "\n\n" + BUG_FOUND_MSG + ":\n\n" + str(e)

    return output, hidden_text


def reset_memory(history, memory):
    memory.clear()
    history = []
    return history, history, memory


class ChatWrapper:

    def __init__(self):
        self.lock = Lock()

    def __call__(
            self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain],
            trace_chain: bool, speak_text: bool, talking_head: bool, monologue: bool, express_chain: Optional[LLMChain],
            num_words, formality, anticipation_level, joy_level, trust_level,
            fear_level, surprise_level, sadness_level, disgust_level, anger_level,
            lang_level, translate_to, literary_style, qa_chain, docsearch, use_embeddings
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
            print("inp: " + inp)
            print("trace_chain: ", trace_chain)
            print("speak_text: ", speak_text)
            print("talking_head: ", talking_head)
            print("monologue: ", monologue)
            history = history or []
            # If chain is None, that is because no API key was provided.
            output = "Please paste your OpenAI key from openai.com to use this app. " + str(datetime.datetime.now())
            hidden_text = output

            if chain:
                # Set OpenAI key
                import openai
                openai.api_key = api_key
                if not monologue:
                    if use_embeddings:
                        if inp and inp.strip() != "":
                            if docsearch:
                                docs = docsearch.similarity_search(inp)
                                output = str(qa_chain.run(input_documents=docs, question=inp))
                            else:
                                output, hidden_text = "Please supply some text in the the Embeddings tab.", None
                        else:
                            output, hidden_text = "What's on your mind?", None
                    else:
                        output, hidden_text = run_chain(chain, inp, capture_hidden_text=trace_chain)
                else:
                    output, hidden_text = inp, None

            output = transform_text(output, express_chain, num_words, formality, anticipation_level, joy_level,
                                    trust_level,
                                    fear_level, surprise_level, sadness_level, disgust_level, anger_level,
                                    lang_level, translate_to, literary_style)

            text_to_display = output
            if trace_chain:
                text_to_display = hidden_text + "\n\n" + output
            history.append((inp, text_to_display))

            html_video, temp_file, html_audio, temp_aud_file = None, None, None, None
            if speak_text:
                if talking_head:
                    if len(output) <= MAX_TALKING_HEAD_TEXT_LENGTH:
                        html_video, temp_file = do_html_video_speak(output, translate_to)
                    else:
                        temp_file = LOOPING_TALKING_HEAD
                        html_video = create_html_video(temp_file, TALKING_HEAD_WIDTH)
                        html_audio, temp_aud_file = do_html_audio_speak(output, translate_to)
                else:
                    html_audio, temp_aud_file = do_html_audio_speak(output, translate_to)
            else:
                if talking_head:
                    temp_file = LOOPING_TALKING_HEAD
                    html_video = create_html_video(temp_file, TALKING_HEAD_WIDTH)
                else:
                    # html_audio, temp_aud_file = do_html_audio_speak(output, translate_to)
                    # html_video = create_html_video(temp_file, "128")
                    pass

        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history, html_video, temp_file, html_audio, temp_aud_file, ""
        # return history, history, html_audio, temp_aud_file, ""


chat = ChatWrapper()


def do_html_audio_speak(words_to_speak, polly_language):
    polly_client = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ["AWS_DEFAULT_REGION"]
    ).client('polly')

    # voice_id, language_code, engine = POLLY_VOICE_DATA.get_voice(polly_language, "Female")
    voice_id, language_code, engine = POLLY_VOICE_DATA.get_voice(polly_language, "Male")
    if not voice_id:
        # voice_id = "Joanna"
        voice_id = "Matthew"
        language_code = "en-US"
        engine = NEURAL_ENGINE
    response = polly_client.synthesize_speech(
        Text=words_to_speak,
        OutputFormat='mp3',
        VoiceId=voice_id,
        LanguageCode=language_code,
        Engine=engine
    )

    html_audio = '<pre>no audio</pre>'

    # Save the audio stream returned by Amazon Polly on Lambda's temp directory
    if "AudioStream" in response:
        with closing(response["AudioStream"]) as stream:
            # output = os.path.join("/tmp/", "speech.mp3")

            try:
                with open('audios/tempfile.mp3', 'wb') as f:
                    f.write(stream.read())
                temp_aud_file = gr.File("audios/tempfile.mp3")
                temp_aud_file_url = "/file=" + temp_aud_file.value['name']
                html_audio = f'<audio autoplay><source src={temp_aud_file_url} type="audio/mp3"></audio>'
            except IOError as error:
                # Could not write to file, exit gracefully
                print(error)
                return None, None
    else:
        # The response didn't contain audio data, exit gracefully
        print("Could not stream audio")
        return None, None

    return html_audio, "audios/tempfile.mp3"


def create_html_video(file_name, width):
    temp_file_url = "/file=" + tmp_file.value['name']
    html_video = f'<video width={width} height={width} autoplay muted loop><source src={temp_file_url} type="video/mp4" poster="Masahiro.png"></video>'
    return html_video


def do_html_video_speak(words_to_speak, azure_language):
    azure_voice = AZURE_VOICE_DATA.get_voice(azure_language, "Male")
    if not azure_voice:
        azure_voice = "en-US-ChristopherNeural"

    headers = {"Authorization": f"Bearer {os.environ['EXHUMAN_API_KEY']}"}
    body = {
        'bot_name': 'Masahiro',
        'bot_response': words_to_speak,
        'azure_voice': azure_voice,
        'azure_style': 'friendly',
        'animation_pipeline': 'high_speed',
    }
    api_endpoint = "https://api.exh.ai/animations/v1/generate_lipsync"
    res = requests.post(api_endpoint, json=body, headers=headers)
    print("res.status_code: ", res.status_code)

    html_video = '<pre>no video</pre>'
    if isinstance(res.content, bytes):
        response_stream = io.BytesIO(res.content)
        print("len(res.content)): ", len(res.content))

        with open('videos/tempfile.mp4', 'wb') as f:
            f.write(response_stream.read())
        temp_file = gr.File("videos/tempfile.mp4")
        temp_file_url = "/file=" + temp_file.value['name']
        html_video = f'<video width={TALKING_HEAD_WIDTH} height={TALKING_HEAD_WIDTH} autoplay><source src={temp_file_url} type="video/mp4" poster="Masahiro.png"></video>'
    else:
        print('video url unknown')
    return html_video, "videos/tempfile.mp4"


def update_selected_tools(widget, state, llm):
    if widget:
        state = widget
        chain, express_chain, memory = load_chain(state, llm)
        return state, llm, chain, express_chain


def update_talking_head(widget, state):
    if widget:
        state = widget

        video_html_talking_head = create_html_video(LOOPING_TALKING_HEAD, TALKING_HEAD_WIDTH)
        return state, video_html_talking_head
    else:
        # return state, create_html_video(LOOPING_TALKING_HEAD, "32")
        return None, "<pre></pre>"


def update_foo(widget, state):
    if widget:
        state = widget
        return state


# Pertains to question answering functionality
def update_embeddings(embeddings_text, embeddings, qa_chain):
    if embeddings_text:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(embeddings_text)

        docsearch = FAISS.from_texts(texts, embeddings)
        print("Embeddings updated")
        return docsearch


# Pertains to question answering functionality
def update_use_embeddings(widget, state):
    if widget:
        state = widget
        return state


with gr.Blocks(css=".gradio-container {background-color: lightgray}") as block:
    llm_state = gr.State()
    history_state = gr.State()
    chain_state = gr.State()
    express_chain_state = gr.State()
    tools_list_state = gr.State(TOOLS_DEFAULT_LIST)
    trace_chain_state = gr.State(False)
    speak_text_state = gr.State(False)
    talking_head_state = gr.State(True)
    monologue_state = gr.State(False)  # Takes the input and repeats it back to the user, optionally transforming it.
    memory_state = gr.State()

    # Pertains to Express-inator functionality
    num_words_state = gr.State(NUM_WORDS_DEFAULT)
    formality_state = gr.State(FORMALITY_DEFAULT)
    anticipation_level_state = gr.State(EMOTION_DEFAULT)
    joy_level_state = gr.State(EMOTION_DEFAULT)
    trust_level_state = gr.State(EMOTION_DEFAULT)
    fear_level_state = gr.State(EMOTION_DEFAULT)
    surprise_level_state = gr.State(EMOTION_DEFAULT)
    sadness_level_state = gr.State(EMOTION_DEFAULT)
    disgust_level_state = gr.State(EMOTION_DEFAULT)
    anger_level_state = gr.State(EMOTION_DEFAULT)
    lang_level_state = gr.State(LANG_LEVEL_DEFAULT)
    translate_to_state = gr.State(TRANSLATE_TO_DEFAULT)
    literary_style_state = gr.State(LITERARY_STYLE_DEFAULT)

    # Pertains to WHISPER functionality
    whisper_lang_state = gr.State(WHISPER_DETECT_LANG)

    # Pertains to question answering functionality
    embeddings_state = gr.State()
    qa_chain_state = gr.State()
    docsearch_state = gr.State()
    use_embeddings_state = gr.State(False)

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column():
                gr.HTML(
                    """<b><center>GPT + WolframAlpha + Whisper</center></b>
                    <p><center>Hit Enter after pasting your OpenAI API key</center></p>""")

            openai_api_key_textbox = gr.Textbox(placeholder="Paste your OpenAI API key (sk-...) and hit Enter",
                                                show_label=False, lines=1, type='password')

        with gr.Row():
            with gr.Column(scale=1, min_width=TALKING_HEAD_WIDTH, visible=True):
                speak_text_cb = gr.Checkbox(label="Enable speech", value=False)
                speak_text_cb.change(update_foo, inputs=[speak_text_cb, speak_text_state],
                                     outputs=[speak_text_state])

                my_file = gr.File(label="Upload a file", type="file", visible=False)
                tmp_file = gr.File(LOOPING_TALKING_HEAD, visible=False)
                # tmp_file_url = "/file=" + tmp_file.value['name']
                htm_video = create_html_video(LOOPING_TALKING_HEAD, TALKING_HEAD_WIDTH)
                video_html = gr.HTML(htm_video)

                # my_aud_file = gr.File(label="Audio file", type="file", visible=True)
                tmp_aud_file = gr.File("audios/tempfile.mp3", visible=False)
                tmp_aud_file_url = "/file=" + tmp_aud_file.value['name']
                htm_audio = f'<audio><source src={tmp_aud_file_url} type="audio/mp3"></audio>'
                audio_html = gr.HTML(htm_audio)

            with gr.Column(scale=7):
                chatbot = gr.Chatbot()

        with gr.Row():
            message = gr.Textbox(label="What's on your mind??",
                                 placeholder="What's the answer to life, the universe, and everything?",
                                 lines=1)
            submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

        # UNCOMMENT TO USE WHISPER
        with gr.Row():
            audio_comp = gr.Microphone(source="microphone", type="filepath", label="Just say it!",
                                       interactive=True, streaming=False)
            audio_comp.change(transcribe, inputs=[audio_comp, whisper_lang_state], outputs=[message])

        # TEMPORARY FOR TESTING
        # with gr.Row():
        #     audio_comp_tb = gr.Textbox(label="Just say it!", lines=1)
        #     audio_comp_tb.submit(transcribe_dummy, inputs=[audio_comp_tb, whisper_lang_state], outputs=[message])

        gr.Examples(
            examples=["How many people live in Canada?",
                      "What is 2 to the 30th power?",
                      "If x+y=10 and x-y=4, what are x and y?",
                      "How much did it rain in SF today?",
                      "Get me information about the movie 'Avatar'",
                      "What are the top tech headlines in the US?",
                      "On the desk, you see two blue booklets, two purple booklets, and two yellow pairs of sunglasses - "
                      "if I remove all the pairs of sunglasses from the desk, how many purple items remain on it?"],
            inputs=message
        )

    with gr.Tab("Settings"):
        tools_cb_group = gr.CheckboxGroup(label="Tools:", choices=TOOLS_LIST,
                                          value=TOOLS_DEFAULT_LIST)
        tools_cb_group.change(update_selected_tools,
                              inputs=[tools_cb_group, tools_list_state, llm_state],
                              outputs=[tools_list_state, llm_state, chain_state, express_chain_state])

        trace_chain_cb = gr.Checkbox(label="Show reasoning chain in chat bubble", value=False)
        trace_chain_cb.change(update_foo, inputs=[trace_chain_cb, trace_chain_state],
                              outputs=[trace_chain_state])

        # speak_text_cb = gr.Checkbox(label="Speak text from agent", value=False)
        # speak_text_cb.change(update_foo, inputs=[speak_text_cb, speak_text_state],
        #                      outputs=[speak_text_state])

        talking_head_cb = gr.Checkbox(label="Show talking head", value=True)
        talking_head_cb.change(update_talking_head, inputs=[talking_head_cb, talking_head_state],
                               outputs=[talking_head_state, video_html])

        monologue_cb = gr.Checkbox(label="Babel fish mode (translate/restate what you enter, no conversational agent)",
                                   value=False)
        monologue_cb.change(update_foo, inputs=[monologue_cb, monologue_state],
                            outputs=[monologue_state])

        reset_btn = gr.Button(value="Reset chat", variant="secondary").style(full_width=False)
        reset_btn.click(reset_memory, inputs=[history_state, memory_state], outputs=[chatbot, history_state, memory_state])

    with gr.Tab("Whisper STT"):
        whisper_lang_radio = gr.Radio(label="Whisper speech-to-text language:", choices=[
            WHISPER_DETECT_LANG, "Arabic", "Arabic (Gulf)", "Catalan", "Chinese (Cantonese)", "Chinese (Mandarin)",
            "Danish", "Dutch", "English (Australian)", "English (British)", "English (Indian)", "English (New Zealand)",
            "English (South African)", "English (US)", "English (Welsh)", "Finnish", "French", "French (Canadian)",
            "German", "German (Austrian)", "Georgian", "Hindi", "Icelandic", "Indonesian", "Italian", "Japanese",
            "Korean", "Norwegian", "Polish",
            "Portuguese (Brazilian)", "Portuguese (European)", "Romanian", "Russian", "Spanish (European)",
            "Spanish (Mexican)", "Spanish (US)", "Swedish", "Turkish", "Ukrainian", "Welsh"],
                                      value=WHISPER_DETECT_LANG)

        whisper_lang_radio.change(update_foo,
                                  inputs=[whisper_lang_radio, whisper_lang_state],
                                  outputs=[whisper_lang_state])

    with gr.Tab("Translate to"):
        lang_level_radio = gr.Radio(label="Language level:", choices=[
            LANG_LEVEL_DEFAULT, "1st grade", "2nd grade", "3rd grade", "4th grade", "5th grade", "6th grade",
            "7th grade", "8th grade", "9th grade", "10th grade", "11th grade", "12th grade", "University"],
                                    value=LANG_LEVEL_DEFAULT)
        lang_level_radio.change(update_foo, inputs=[lang_level_radio, lang_level_state],
                                outputs=[lang_level_state])

        translate_to_radio = gr.Radio(label="Language:", choices=[
            TRANSLATE_TO_DEFAULT, "Arabic", "Arabic (Gulf)", "Catalan", "Chinese (Cantonese)", "Chinese (Mandarin)",
            "Danish", "Dutch", "English (Australian)", "English (British)", "English (Indian)", "English (New Zealand)",
            "English (South African)", "English (US)", "English (Welsh)", "Finnish", "French", "French (Canadian)",
            "German", "German (Austrian)", "Georgian", "Hindi", "Icelandic", "Indonesian", "Italian", "Japanese",
            "Korean", "Norwegian", "Polish",
            "Portuguese (Brazilian)", "Portuguese (European)", "Romanian", "Russian", "Spanish (European)",
            "Spanish (Mexican)", "Spanish (US)", "Swedish", "Turkish", "Ukrainian", "Welsh",
            "emojis", "Gen Z slang", "how the stereotypical Karen would say it", "Klingon", "Neanderthal",
            "Pirate", "Strange Planet expospeak technical talk", "Yoda"],
                                      value=TRANSLATE_TO_DEFAULT)

        translate_to_radio.change(update_foo,
                                  inputs=[translate_to_radio, translate_to_state],
                                  outputs=[translate_to_state])

    with gr.Tab("Formality"):
        formality_radio = gr.Radio(label="Formality:",
                                   choices=[FORMALITY_DEFAULT, "Casual", "Polite", "Honorific"],
                                   value=FORMALITY_DEFAULT)
        formality_radio.change(update_foo,
                               inputs=[formality_radio, formality_state],
                               outputs=[formality_state])

    with gr.Tab("Lit style"):
        literary_style_radio = gr.Radio(label="Literary style:", choices=[
            LITERARY_STYLE_DEFAULT, "Prose", "Story", "Summary", "Outline", "Bullets", "Poetry", "Haiku", "Limerick", "Rap",
            "Joke", "Knock-knock", "FAQ"],
                                        value=LITERARY_STYLE_DEFAULT)

        literary_style_radio.change(update_foo,
                                    inputs=[literary_style_radio, literary_style_state],
                                    outputs=[literary_style_state])

    with gr.Tab("Emotions"):
        anticipation_level_radio = gr.Radio(label="Anticipation level:",
                                            choices=[EMOTION_DEFAULT, "Interest", "Anticipation", "Vigilance"],
                                            value=EMOTION_DEFAULT)
        anticipation_level_radio.change(update_foo,
                                        inputs=[anticipation_level_radio, anticipation_level_state],
                                        outputs=[anticipation_level_state])

        joy_level_radio = gr.Radio(label="Joy level:",
                                   choices=[EMOTION_DEFAULT, "Serenity", "Joy", "Ecstasy"],
                                   value=EMOTION_DEFAULT)
        joy_level_radio.change(update_foo,
                               inputs=[joy_level_radio, joy_level_state],
                               outputs=[joy_level_state])

        trust_level_radio = gr.Radio(label="Trust level:",
                                     choices=[EMOTION_DEFAULT, "Acceptance", "Trust", "Admiration"],
                                     value=EMOTION_DEFAULT)
        trust_level_radio.change(update_foo,
                                 inputs=[trust_level_radio, trust_level_state],
                                 outputs=[trust_level_state])

        fear_level_radio = gr.Radio(label="Fear level:",
                                    choices=[EMOTION_DEFAULT, "Apprehension", "Fear", "Terror"],
                                    value=EMOTION_DEFAULT)
        fear_level_radio.change(update_foo,
                                inputs=[fear_level_radio, fear_level_state],
                                outputs=[fear_level_state])

        surprise_level_radio = gr.Radio(label="Surprise level:",
                                        choices=[EMOTION_DEFAULT, "Distraction", "Surprise", "Amazement"],
                                        value=EMOTION_DEFAULT)
        surprise_level_radio.change(update_foo,
                                    inputs=[surprise_level_radio, surprise_level_state],
                                    outputs=[surprise_level_state])

        sadness_level_radio = gr.Radio(label="Sadness level:",
                                       choices=[EMOTION_DEFAULT, "Pensiveness", "Sadness", "Grief"],
                                       value=EMOTION_DEFAULT)
        sadness_level_radio.change(update_foo,
                                   inputs=[sadness_level_radio, sadness_level_state],
                                   outputs=[sadness_level_state])

        disgust_level_radio = gr.Radio(label="Disgust level:",
                                       choices=[EMOTION_DEFAULT, "Boredom", "Disgust", "Loathing"],
                                       value=EMOTION_DEFAULT)
        disgust_level_radio.change(update_foo,
                                   inputs=[disgust_level_radio, disgust_level_state],
                                   outputs=[disgust_level_state])

        anger_level_radio = gr.Radio(label="Anger level:",
                                     choices=[EMOTION_DEFAULT, "Annoyance", "Anger", "Rage"],
                                     value=EMOTION_DEFAULT)
        anger_level_radio.change(update_foo,
                                 inputs=[anger_level_radio, anger_level_state],
                                 outputs=[anger_level_state])

    with gr.Tab("Max words"):
        num_words_slider = gr.Slider(label="Max number of words to generate (0 for don't care)",
                                     value=NUM_WORDS_DEFAULT, minimum=0, maximum=MAX_WORDS, step=10)
        num_words_slider.change(update_foo,
                                inputs=[num_words_slider, num_words_state],
                                outputs=[num_words_state])

    with gr.Tab("Embeddings"):
        embeddings_text_box = gr.Textbox(label="Enter text for embeddings and hit Create:",
                                         lines=20)

        with gr.Row():
            use_embeddings_cb = gr.Checkbox(label="Use embeddings", value=False)
            use_embeddings_cb.change(update_use_embeddings, inputs=[use_embeddings_cb, use_embeddings_state],
                                     outputs=[use_embeddings_state])

            embeddings_text_submit = gr.Button(value="Create", variant="secondary").style(full_width=False)
            embeddings_text_submit.click(update_embeddings,
                                         inputs=[embeddings_text_box, embeddings_state, qa_chain_state],
                                         outputs=[docsearch_state])

    gr.HTML("""
        <p>This application, developed by <a href='https://www.linkedin.com/in/javafxpert/'>James L. Weaver</a>, 
        demonstrates a conversational agent implemented with OpenAI GPT-3.5 and LangChain. 
        When necessary, it leverages tools for complex math, searching the internet, and accessing news and weather.
        Uses talking heads from <a href='https://exh.ai/'>Ex-Human</a>.
        For faster inference without waiting in queue, you may duplicate the space.
        </p>""")

    gr.HTML("""
<form action="https://www.paypal.com/donate" method="post" target="_blank">
<input type="hidden" name="business" value="AK8BVNALBXSPQ" />
<input type="hidden" name="no_recurring" value="0" />
<input type="hidden" name="item_name" value="Please consider helping to defray the cost of APIs such as SerpAPI and WolframAlpha that this app uses." />
<input type="hidden" name="currency_code" value="USD" />
<input type="image" src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif" border="0" name="submit" title="PayPal - The safer, easier way to pay online!" alt="Donate with PayPal button" />
<img alt="" border="0" src="https://www.paypal.com/en_US/i/scr/pixel.gif" width="1" height="1" />
</form>
    """)

    gr.HTML("""<center>
        <a href="https://huggingface.co/spaces/JavaFXpert/Chat-GPT-LangChain?duplicate=true">
        <img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a>
        </center>""")

    message.submit(chat, inputs=[openai_api_key_textbox, message, history_state, chain_state, trace_chain_state,
                                 speak_text_state, talking_head_state, monologue_state,
                                 express_chain_state, num_words_state, formality_state,
                                 anticipation_level_state, joy_level_state, trust_level_state, fear_level_state,
                                 surprise_level_state, sadness_level_state, disgust_level_state, anger_level_state,
                                 lang_level_state, translate_to_state, literary_style_state,
                                 qa_chain_state, docsearch_state, use_embeddings_state],
                   outputs=[chatbot, history_state, video_html, my_file, audio_html, tmp_aud_file, message])
    # outputs=[chatbot, history_state, audio_html, tmp_aud_file, message])

    submit.click(chat, inputs=[openai_api_key_textbox, message, history_state, chain_state, trace_chain_state,
                               speak_text_state, talking_head_state, monologue_state,
                               express_chain_state, num_words_state, formality_state,
                               anticipation_level_state, joy_level_state, trust_level_state, fear_level_state,
                               surprise_level_state, sadness_level_state, disgust_level_state, anger_level_state,
                               lang_level_state, translate_to_state, literary_style_state,
                               qa_chain_state, docsearch_state, use_embeddings_state],
                 outputs=[chatbot, history_state, video_html, my_file, audio_html, tmp_aud_file, message])
    # outputs=[chatbot, history_state, audio_html, tmp_aud_file, message])

    openai_api_key_textbox.change(set_openai_api_key,
                                  inputs=[openai_api_key_textbox],
                                  outputs=[chain_state, express_chain_state, llm_state, embeddings_state,
                                           qa_chain_state, memory_state])
    openai_api_key_textbox.submit(set_openai_api_key,
                                  inputs=[openai_api_key_textbox],
                                  outputs=[chain_state, express_chain_state, llm_state, embeddings_state,
                                           qa_chain_state, memory_state])

block.launch(debug=True)
