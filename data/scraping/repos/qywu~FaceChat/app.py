async_mode = None

if async_mode is None:
    try:
        import eventlet

        async_mode = "eventlet"
    except ImportError:
        pass

    if async_mode is None:
        try:
            from gevent import monkey

            async_mode = "gevent"
        except ImportError:
            pass

    if async_mode is None:
        async_mode = "threading"

    print("async_mode is " + async_mode)

# monkey patching is necessary because this application uses a background
# thread
if async_mode == "eventlet":
    import eventlet

    eventlet.monkey_patch()
elif async_mode == "gevent":
    from gevent import monkey

    monkey.patch_all()
# The Session instance is not used for direct access, you should always use flask.session
from flask_session import Session
import os
import random
import torch
import time
from threading import Thread
import collections
import queue
from flask import Flask, render_template, request, session, redirect
from flask_socketio import SocketIO
import numpy as np
import threading
#DEEMA#
from pydub import AudioSegment
sem = threading.Semaphore(1)

# audio processing
# from transformers import AutoProcessor, WhisperForConditionalGeneration
import scipy.signal as sps
import webrtcvad
from transformers import AutoProcessor, WhisperForConditionalGeneration
from TTS.api import TTS
import openai
from flask_session import Session

# image processing
import base64, cv2
import io
from PIL import Image
from engineio.payload import Payload
from deepface import DeepFace
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

Payload.max_decode_packets = 2048
openai.api_key = os.environ["OPENAI_API_KEY"]

app = Flask(__name__)
app.debug = True

socketio = SocketIO(app, cors_allowed_origins="*")
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

def isEnglish(s):
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def genWaveHeader(sampleRate, bitsPerSample, channels, samples):
    datasize = samples * channels * bitsPerSample // 8
    o = bytes("RIFF", "ascii")  # (4byte) Marks file as RIFF
    o += (datasize + 36).to_bytes(4, "little")  # (4byte) File size in bytes excluding this and RIFF marker
    o += bytes("WAVE", "ascii")  # (4byte) File type
    o += bytes("fmt ", "ascii")  # (4byte) Format Chunk Marker
    o += (16).to_bytes(4, "little")  # (4byte) Length of above format data
    o += (1).to_bytes(2, "little")  # (2byte) Format type (1 - PCM)
    o += (channels).to_bytes(2, "little")  # (2byte)
    o += (sampleRate).to_bytes(4, "little")  # (4byte)
    o += (sampleRate * channels * bitsPerSample // 8).to_bytes(4, "little")  # (4byte)
    o += (channels * bitsPerSample // 8).to_bytes(2, "little")  # (2byte)
    o += (bitsPerSample).to_bytes(2, "little")  # (2byte)
    o += bytes("data", "ascii")  # (4byte) Data Chunk Marker
    o += (datasize).to_bytes(4, "little")  # (4byte) Data size in bytes
    return o


def Int2Float(sound):
    _sound = np.copy(sound)  #
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype("float32")
    if abs_max > 0:
        _sound *= 1 / abs_max
    audio_float32 = torch.from_numpy(_sound.squeeze())
    return audio_float32


class ASR:
    def __init__(self) -> None:
        #**BENCHMARK**#
        # it was model_name
        #self.model_name = "openai/whisper-tiny" #1
        #self.model_name = "openai/whisper-base" #2
        #self.model_name = "openai/whisper-small" #3
        self.model_name = "openai/whisper-medium" #4
        #self.model_name = "openai/whisper-large" #5

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        self.model_sample_rate = 16000
        self.device = torch.device("cpu")

    def __call__(self, data, sample_rate=16000) -> str:
        """
        Args:
            data: PCM float32 format
            sample_rate: the sample rate of data
        """
        is_valid = True
        # first, resample the data to the model's sample_rate
        if sample_rate != self.model_sample_rate:
            number_of_samples = round(len(data) * float(self.model_sample_rate) / sample_rate)
            data = sps.resample(data, number_of_samples)

        # genearte text
        inputs = self.processor(data, return_tensors="pt", sampling_rate=self.model_sample_rate)
        input_features = inputs.input_features.to(self.device)
        generated_ids = self.model.generate(inputs=input_features)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if len(generated_ids[0]) < 4:
            is_valid = False

        if not isEnglish(text):
            is_valid = False

        return text, is_valid

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self


class TTSModel:
    def __init__(self) -> None:
        self.model = TTS("tts_models/en/vctk/vits", gpu=True)

    def __call__(self, text) -> np.float32:
        wav = self.model.tts(text, speaker=self.model.speakers[33])
        return wav


class AudioContext:
    """Streams raw audio from web microphone. Data is received in a separate thread, and stored in a buffer, to be read from.
    """

    MIC_RATE = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self):
        self.audio_buffer = queue.Queue()

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.audio_buffer.get()

    def update(self, audioData):
        """Update the audio buffer."""
        self.audio_buffer.put(audioData)


class GPT3Chatbot:
    def __init__(self, model_name="text-davinci-003"):
        self.model_name = model_name
        self.instruct_prompt = \
                """Agent: Hello, How can I help you today?
    """
   # """Background: Sony Life Insurance Co., Ltd. is a Japanese insurance company founded in 1979 as a joint venture between Sony Corporation and Prudential Financial and headquartered in Tokyo.
   # Instruction: Be an insurance agent that works for Sony Insurance Company and chat with a customer. The response should be short and informative.

    #Agent: Hello, I am an insurance agent from Sony Insurance Company. How can I help you today?
   # """

        self.bot_prompt = "Agent: "
        self.user_prompt = "User: "
        self.context = self.instruct_prompt
        self.emotion_prompt = "(User is in neutral emotion)"

        if "davinci-003" in model_name:
            self.use_emotion_prompt = True
        else:
            self.use_emotion_prompt = False

    def get_response(self, user_input):
        if "restart" in user_input.lower() or "reset" in user_input.lower():
            self.reset()
            return "Hello, How can I help you today?" #"Hello, I am an insurance agent from Sony Insurance Company. How can I help you today?"

        error_responses = ["Let me think...", "Give me some seconds...", "Wait a second"]
        user_input = self.user_prompt + user_input + "\n"
        if self.use_emotion_prompt:
            user_input += self.emotion_prompt + "\n"
        completion_prompt = self.context + user_input + self.bot_prompt

        request_success = False
        while not request_success:
            try:
                response = openai.Completion.create(
                    model=self.model_name, prompt=completion_prompt, temperature=0.95, max_tokens=128, top_p=0.95,
                )
                request_success = True
            except Exception as e:
                print(e)
                error_response = random.choice(error_responses)
                audio_speak(error_response)
                print("Request failed, retrying...")
                

        response = response["choices"][0]["text"].strip()

        self.context += user_input + self.bot_prompt + response + "\n"

        return response

    def reset(self):
        self.context = self.instruct_prompt
        reset_audio_buffer()

# GPT models
# to be used for the BENCHMARK

chatbot = GPT3Chatbot("text-davinci-002")
#chatbot = GPT3Chatbot("text-davinci-003")
#chatbot = GPT3Chatbot("text-curie-001")
#chatbot = GPT3Chatbot("text-babbage-001")
#chatbot = GPT3Chatbot("text-ada-001")


asr_model = ASR()
tts_model = TTSModel()
# specify the running device
# device = torch.device("cuda")
# force cuda
device = torch.device('cuda')

asr_model = asr_model.to(device)

audio_buffer = queue.Queue()

audio_buffer_lock = False


def reset_audio_buffer():
    global audio_buffer
    audio_buffer.queue.clear()


@socketio.on("audio_listen")
def audio_listen(audioData):
    global audio_context
    global audio_buffer
    global audio_buffer_lock

    if not audio_buffer_lock:
        audio_buffer.put(audioData)


@socketio.on("start_chat")
def start_chat(data):
    global audio_buffer_lock
    audio_buffer_lock = False


@socketio.on("stop_chat")
def stop_chat(data):
    print("stopping...")
    global audio_buffer_lock
    audio_buffer_lock = True
    session["name"] = None
    sem.release()
    socketio.emit('logout', "login");
    # to be used for the BENCHMARK
    f.close()

@socketio.on("system_init")
def system_init(audioData):
    # speak
    #audio_speak("Hello, I am an insurance agent from Sony Insurance Company. How can I help you today?")
    audio_speak("Hello, How can I help you today?")
    # # delay the next request


@app.route("/update-text", methods=["POST"])
def update_text():
    text = chatbot.context
    return text


@app.route("/", methods=["POST", "GET"])
def index():
    print("intialized")
    if not session.get("name"):
        return redirect("/login")
    if sem._value == 0:
        return render_template("login.html", msg="Please wait, the agent is busy with another client...")
    sem.acquire()
    # reset the chatbot and buffer queue
    chatbot.reset()
    reset_audio_buffer()
    return render_template("index.html")

@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method=='POST':
        session["name"] = request.form.get('name')
        return redirect("/")
    return render_template("login.html")
 
class VADAudio:
    """Filter & segment audio with voice activity detection."""

    def __init__(self, input_rate, audio_context):
        self.input_rate = input_rate
        self.audio_context = audio_context
        self.RATE_PROCESS = 16000
        self.block_size = 743
        self.frame_duration_ms = 1000 * self.block_size // self.input_rate
        self.sample_rate = 16000
        self.silence_duration_ms = 500

        self.vad = webrtcvad.Vad(mode=3)

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        global vad_model
        global vad_iterator
        global audio_buffer
        global audio_buffer_lock
        
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        empty_frame_count = 0
        max_empty_frame_count = self.silence_duration_ms // self.frame_duration_ms

        while True:
            if audio_buffer_lock:
                continue

            frame = audio_buffer.get()

            is_speech = self.vad.is_speech(frame[-960:], self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                # if speaking
                num_voiced = len([f for f, is_speech in ring_buffer if is_speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for frame, is_speech in ring_buffer:
                        yield frame
                    ring_buffer.clear()
            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                # if not seapking
                num_unvoiced = len([f for f, is_speech in ring_buffer if not is_speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    # detects 5 consecutive empty frames
                    if empty_frame_count > max_empty_frame_count:
                        triggered = False
                        yield None
                        ring_buffer.clear()
                        empty_frame_count = 0
                    else:
                        empty_frame_count += 1
                else:
                    # reset empty_frame_count if detects speech
                    empty_frame_count = 0

# to be used for the BENCHMARK
f = open(str(chatbot.model_name)+"_"+str(asr_model.model_name[asr_model.model_name.index("/")+1:])+".txt", "a")

class EngagementDetector(Thread):
    def __init__(self, audio_context):
        Thread.__init__(self)
        self.audio_context = audio_context
        self.vad_audio = VADAudio(input_rate=16000, audio_context=self.audio_context)
        self.vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        (self.get_speech_ts, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils
        self.count = 0
            

    def run(self):
        frames = self.vad_audio.vad_collector()
        wav_data = bytearray()

        vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        (get_speech_ts, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils

        print("Listening...")

        for frame in frames:
            
            if frame is not None:
                wav_data.extend(frame)
            else:
                data = np.frombuffer(wav_data, np.int16)
                data = Int2Float(data)

                # two-stage VAD
                time_stamps = get_speech_ts(data, vad_model)

                if len(time_stamps) > 0:
                    print("Speaking:", end="")
                    # to be used for the BENCHMARK
                    s_time = time.time()
                    text, is_asr_valid = asr_model(data, sample_rate=16000)
                    print(text)

                    if is_asr_valid:
                        chatbot_response = chatbot.get_response(text)
                        # to be used for the BENCHMARK
                        f.write(str(time.time()-s_time))
                        f.write("\n")
                        # speak
                        audio_speak(chatbot_response)

                    # clear buffer if speech detected
                    wav_data = bytearray()


def audio_speak(text):
    global audio_buffer_lock
    print(text)
    audio_buffer_lock = True
    # reset audio buffer to avoid interference
    reset_audio_buffer()
    audio_float32 = tts_model(text)
    audio_int16 = (np.array(audio_float32, dtype=np.float32) * 32768).astype(np.int16)

    wav_header = genWaveHeader(sampleRate=22050, bitsPerSample=16, channels=1, samples=len(audio_int16))

    speak_data = wav_header + audio_int16.tobytes()
    now = len(text.split(" "))

    audio_io = io.BytesIO(speak_data)

    # Create AudioSegment object from the in-memory file object
    # Get duration of audio in milliseconds
    duration_ms = AudioSegment.from_file(audio_io, format="wav")

    duration_ms = len(duration_ms)
    
    # Convert duration to seconds
    now = duration_ms/1000.0

    # we need the size of the text
    print(f"TTS Duration: {now}")
    socketio.emit("audio_speak",  {"voice": speak_data, "words": now});
    print(f"sending data! {text}")

    time.sleep(len(audio_int16) / 22050)
    audio_buffer_lock = False



def read_image_b64(base64_string):
    "decode base64"
    idx = base64_string.find("base64,")
    base64_string = base64_string[idx + 7 :]
    sbuf = io.BytesIO()
    sbuf.write(base64.b64decode(base64_string, " /"))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def moving_average(x):
    return np.mean(x)


# given 20 fps, control the image buffer
image_buffer = queue.Queue(maxsize=5)


@socketio.on("image_observe")
def image_observe(data_image):
    global image_buffer
    frame = read_image_b64(data_image)
    image_buffer.put(frame)


class VideoProcessor(Thread):
    def __init__(self, image_buffer):
        Thread.__init__(self)
        self.image_buffer = image_buffer
        self._fps_array = [0]

    def frame_generator(self):
        while True:
            frame = self.image_buffer.get()
            yield frame

    def run(self):
        frames = self.frame_generator()
        prev_recv_time = time.time()
        fps = 0
        cnt = 0

        prev_box_pos = np.array([0, 0, 0, 0])
        prev_scores = np.zeros(7)
        emotion_beta = 0.99
        box_beta = 0.2

        for frame in frames:
            try:
                obj = DeepFace.analyze(
                    frame, actions=["emotion"], enforce_detection=False, silent=True, detector_backend="ssd"
                )

                if isinstance(obj, list):
                    obj = obj[0]
            except Exception as e:
                print(e)
                continue

            emotions, scores = zip(*obj["emotion"].items())
            scores = list(scores)
            # emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            # mask out disgust
            scores[1] = scores[1] - 1000
            # mask out fear
            scores[2] = scores[2] - 1000
            # mask out surprise
            scores[5] = scores[5] - 1000
            # give more weight to happy
            # scores[3] = scores[3] * 2.0

            scores = prev_scores * emotion_beta + np.array(scores) * (1 - emotion_beta)
            # apply softmax
            scores = np.exp(scores) / np.sum(np.exp(scores))
            prev_scores = scores

            index = np.argmax(scores)
            pred_emotion = emotions[index]

            # x, y, w, h
            box_pos = np.array(list(obj["region"].values()))

            if (
                pred_emotion in emotions
                and (box_pos[0] > 0 and box_pos[1] > 0)
                and (box_pos[0] < 400 and box_pos[1] < 300)
            ):
                box_pos = prev_box_pos * box_beta + box_pos * (1 - box_beta)
                box_pos = np.rint(box_pos).astype(int)
                x, y, w, h = box_pos
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    pred_emotion,
                    (x - 10, y - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.9,
                    color=(255, 0, 0),
                    thickness=2,
                )
                # old_pred_emotion = pred_emotion
                prev_box_pos = box_pos

                if pred_emotion == "happy":
                    chatbot.emotion_prompt = "(User is in happy emotion)"
                elif pred_emotion == "sad":
                    chatbot.emotion_prompt = "(User is in sad emotion)"
                elif pred_emotion == "angry":
                    chatbot.emotion_prompt = "(User is in angry emotion)"
                elif pred_emotion == "neutral":
                    chatbot.emotion_prompt = "(User is in neutral emotion)"
            else:
                pred_emotion = "neutral"
                chatbot.emotion_prompt = "(User is in neutral emotion)"

            recv_time = time.time()
            cv2.putText(
                frame,
                "fps " + str(fps),
                (10, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 0, 0),
                thickness=1,
            )

            # encode it into jpeg
            imgencode = cv2.imencode(".jpeg", frame, [cv2.IMWRITE_JPEG_QUALITY, 40])[1]

            # base64 encode
            stringData = base64.b64encode(imgencode).decode("utf-8")
            b64_src = "data:image/jpeg;base64,"
            stringData = b64_src + stringData

            # emit the frame back
            socketio.emit("image_show", stringData)

            fps = 1 / (recv_time - prev_recv_time)
            self._fps_array.append(fps)
            fps = round(moving_average(np.array(self._fps_array)), 1)
            prev_recv_time = recv_time

            # print(fps_array)
            cnt += 1
            if cnt == 30:
                self._fps_array = [fps]
                cnt = 0


# Globals
audio_context = AudioContext()
engagement_detector = EngagementDetector(audio_context)
engagement_detector.start()

video_process = VideoProcessor(image_buffer)
video_process.start()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=55009, debug=False, keyfile="key.pem", certfile="cert.pem")
    # app.run(host='0.0.0.0', debug=True, threaded=True, port=9900, ssl_context=("cert.pem", "key.pem"))
