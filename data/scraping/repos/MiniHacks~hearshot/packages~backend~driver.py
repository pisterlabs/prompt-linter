import socket
from models import Alert
from copy import deepcopy
import os
from time import sleep, time
import io
from queue import Queue
from tempfile import NamedTemporaryFile
import speech_recognition as sr
import torch
import soundfile as sf
import threading
from transcribe import transcribe
from pathlib import Path
import openai
import json
import requests
import random

from firebase_admin import credentials, firestore, initialize_app
from dotenv import load_dotenv
from models import TranscriptSection, Location, Severity

ENV_PATH = Path(__file__).parent.parent.parent.absolute().joinpath(".env")

load_dotenv(dotenv_path=ENV_PATH)

openai.api_key = os.environ.get("OPENAI_API_KEY")


localhost = "127.0.0.1"
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
MODEL = "gpt-3.5-turbo"

SYSTEM_PROMPT = """You are part of an app to warn people about dangerous situations by extracting information from police radio. You will be given a transcript. Note that it is likely not accurate due to low quality data, use common sense to make changes. Use your expert knowledge of police codes and terminology to extract as many events as possible in the form. Be liberal in your interpretation and make sure to not include polie codes in the "alert" field. Additionally, include a "theories" condensing all facts that are related to unknown events. Cut out useless information. I need you to output a JSON with an event for every address listed in the transcript. Do not include any quotes or slashes. It is crucial that you match this format:

{
    "events": [{
    "alert": "<succint tag line>", (example: "Armed Suspect", "Request for Unit", "Ongoing Investigation") 
    "address": "<human readable address, approximate is fine>", 
    "severity": "low" | "med" | "high", 
    "context": "<relevant transcript>"
}, {...}, ...],

}, {...}, ...],

    "remaining_facts": [str]
}
"""


USE_ONNX = True
silero_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=USE_ONNX,
)
(get_speech_timestamps, _, _, VADIterator, collect_chunks) = utils

# credentias w service account JSON
cred = credentials.Certificate("google_creds.json")

initialize_app(cred)

db = firestore.client()

# This port doesn't matter, it just helps with consistency
audio_file_sender_port = 12345

# only this one does
audio_file_receiver_port = 5555


def convert_to_lat_long(
    address_query: str,
    static_longitude: float = -118.2518,
    static_latitude: float = 34.0488,
) -> Location | None:
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": address_query,  # example: "13702 Hoover Street",
        "fields": "formatted_address,name,geometry",
        "inputtype": "textquery",
        "locationbias": f"circle:50000@{static_latitude},{static_longitude}",
        "key": environ["MAPS_API_KEY"],
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        candidate = data["candidates"][0]
        return Location(
            address=candidate["formatted_address"],
            raw_address=address_query,
            name=candidate["name"],
            latitude=candidate["geometry"]["location"]["lat"],
            longitude=candidate["geometry"]["location"]["lng"],
        )
    else:
        print(f"Error: {response.status_code}")


def complete_chat(
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.98,
    stream=True,
    timeout=30,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}",
    }

    json_data = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "user": "Y2jtYmFpCg==",
    }

    if stream:
        json_data["stream"] = True

        def handle_response(response):
            lines = [line.strip() for line in response.splitlines()]
            lines = [line for line in lines if line != ""]

            completion = ""
            # skip DONE at the end
            for line in lines[:-1]:
                # each line is in the form `data: <json>`
                json_str = line[6:]
                data = json.loads(json_str)
                # structure from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
                choice = data.get("choices", [{}])[0]
                content = choice.get("delta", {}).get("content", "")
                if content:
                    completion += content
            return completion

        try:
            with requests.post(
                "https://api.openai.com/v1/chat/completions",
                json=json_data,
                headers=headers,
                timeout=30,
                stream=True,
            ) as response:
                return handle_response(response.text)
        except Exception as e:
            raise e
    else:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
            timeout=30,
        )
        response_json = response.json()
        completion: str = response_json["choices"][0]["message"]["content"]

        return completion


def complete(
    system_prompt: str | None,
    user_prompt: str | None,
    temperature: float = 0.7,
    top_p: float = 0.98,
    stream=True,
    timeout=30,
    **kwargs,
) -> str:
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt is not None:
        messages.append({"role": "user", "content": user_prompt})
    return complete_chat(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        timeout=timeout,
    )


def read_audio(file, sampling_rate: int = SAMPLE_RATE) -> torch.Tensor:
    file.seek(0)
    audio, _ = sf.read(file, dtype="float32")
    # reshape to (n,)
    return torch.from_numpy(audio).view(-1)


def receive_packets(data_queue):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", audio_file_receiver_port))

    while True:
        audio_data, address = sock.recvfrom(8000)
        data_queue.put(audio_data)


def transcribe_packets(
    data_queue: Queue[bytes], transcript_queue: Queue[TranscriptSection]
):
    curr_section = TranscriptSection(content="", start=None, end=None)
    finished: list[TranscriptSection] = [curr_section]
    temp_file: str = NamedTemporaryFile().name

    last_time = None
    last_sample = bytes()

    while True:
        current_time: float = time()
        if not data_queue.empty():
            phrase_complete = False
            if not curr_section.start:
                curr_section.start = current_time

            if last_time and current_time - last_time > 2:
                last_sample = bytes()
                phrase_complete = True

            last_time = current_time
            while not data_queue.empty():
                data: bytes = data_queue.get()

                audio_data: sr.AudioData = sr.AudioData(data, SAMPLE_RATE, SAMPLE_WIDTH)
                wav_data: io.BytesIO = io.BytesIO(audio_data.get_wav_data())

                # Use Silero VAD to skip segments without voice
                wav: torch.Tensor = read_audio(wav_data, sampling_rate=SAMPLE_RATE)
                # assert wav.shape == (4000,)

                window_size_samples = 512
                speech_probs_and_chunks = [
                    (silero_model(chunk, SAMPLE_RATE).item(), chunk)
                    for chunk in (
                        wav[i : i + window_size_samples]
                        for i in range(0, len(wav), window_size_samples)
                    )
                    if len(chunk) == window_size_samples
                ]
                silero_model.reset_states()

                speech_probs = [prob for prob, _ in speech_probs_and_chunks]
                if max(speech_probs) < 0.3:
                    continue

                last_sample += data

            audio_data = sr.AudioData(last_sample, SAMPLE_RATE, SAMPLE_WIDTH)
            wav_data = io.BytesIO(audio_data.get_wav_data())

            with open(temp_file, "w+b") as f:
                f.write(wav_data.read())

            result = transcribe(temp_file)

            assert type(result["text"]) == str
            text = result["text"].strip()

            if phrase_complete:
                if not text.strip():
                    continue
                curr_section.content = text
                curr_section.end = current_time
                finished[-1] = curr_section
                transcript_queue.put(curr_section)
                print(
                    f"{curr_section.content} ({curr_section.start} - {curr_section.end})"
                )
                curr_section = TranscriptSection(content="", start=None, end=None)
                finished.append(curr_section)
                last_time = None
            else:
                curr_section.content = text
                curr_section.end = current_time
                finished[-1] = curr_section

        sleep(0.25)


def upsert_alert(alert: Alert):
    time, doc = db.collection("alerts").add(
        {
            "label": alert.label,
            "date": alert.date,
            "severity": alert.severity.value,
            "address": alert.address,
            "raw_address": alert.raw_address,
            "name": alert.name,
            "coord": alert.coord,
        }
    )
    print(f"UPSERTED {alert}")


def process_events(transcript_queue: Queue[TranscriptSection]):
    context = ""
    while True:
        if not transcript_queue.empty():
            while transcript_queue:
                item = deepcopy(transcript_queue.get())
                new_section = {
                    "content": item.content,
                    "start": item.start,
                    "end": item.end,
                }

                # Add the new section to the existing Firebase object
                transcription_ref = db.collection("transcriptions").document("lapd")
                transcription = transcription_ref.get()
                sections = transcription.get("sections")
                sections.append(new_section)
                transcription_ref.update({"sections": sections})

                context += f"\n{item.content}"

                try:
                    completion = complete(
                        system_prompt=SYSTEM_PROMPT, user_prompt=context
                    )
                    print(f"{completion=}")
                    d = json.loads(completion)
                    context = "\n".join(d["context"])
                    events = d["events"]
                    alerts = []
                    for event in events:
                        label = event["alert"]
                        date = time()
                        transcript = TranscriptSection(
                            content=event["content"],
                            start=time() - random.randint(20, 30),
                            end=time() - random.randint(5, 15),
                        )

                        raw_address = event["address"]

                        location = convert_to_lat_long(raw_address)

                        if not location:
                            continue

                        coord = (location.latitude, location.longitude)
                        name = location.name
                        address = location.address

                        severity = (
                            Severity.HIGH
                            if d["severity"] == "high"
                            else Severity.LOW
                            if d["severity"] == "low"
                            else Severity.MEDIUM
                        )

                        alert = Alert(
                            label=label,
                            date=date,
                            severity=severity,
                            raw_address=raw_address,
                            transcript=[transcript],
                            address=address,
                            name=name,
                            coord=coord,
                        )

                        upsert_alert(alert)

                except Exception as e:
                    print(e)
                    continue

        sleep(0.25)


def main():
    packet_queue: Queue[bytes] = Queue()
    transcript_queue: Queue[str] = Queue()

    recv_thread = threading.Thread(target=receive_packets, args=(packet_queue,))
    transcribe_thread = threading.Thread(
        target=transcribe_packets, args=(packet_queue, transcript_queue)
    )
    event_thread = threading.Thread(target=process_events, args=(transcript_queue,))

    recv_thread.start()
    transcribe_thread.start()
    event_thread.start()

    recv_thread.join()
    transcribe_thread.join()
    event_thread.join()


if __name__ == "__main__":
    main()
