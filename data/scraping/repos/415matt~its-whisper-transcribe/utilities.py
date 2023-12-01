from datetime import timedelta
import whisper
import openai
import os


def transcribe(filepath, model = "base.en"):
    """
    Transcribe speech from an audio file using a Whisper ASR model.
    """
    model = whisper.load_model(model)

    # parameters have been tuned for long form transcription
    result = model.transcribe(filepath, 
                              language="en",
                              verbose = False, 
                              no_speech_threshold= 0.5, 
                              logprob_threshold = -0.4)
    return result

def query(prompt, excerpt, stream=False):
    """
    Send a chat-based query to the Llama-2-70b chat model and return its response.
    """
    openai.api_base = "https://its-llamarama.ucsd.edu/v1"
    openai.api_key = "12345" # this doesn't matter

    chat_completion = openai.ChatCompletion.create(model="Llama-2-70b-chat", 
                                                   max_tokens= 2000,
                                                   request_timeout=600,
                                                   stream=stream,
                                                   messages=[{"role": "user", "content": prompt + excerpt}])
    
    if stream:
        collected_chunks = []
        collected_messages = ""

        # capture and print event stream
        for chunk in chat_completion:
            collected_chunks.append(chunk)  # save the event response
            chunk_message = chunk['choices'][0]['delta']  # extract the message
            if "content" in chunk_message:
                message_text = chunk_message['content']
                collected_messages += message_text
                print(f"{message_text}", end="")
        print(f"\n")
        return collected_messages


    else:
        return chat_completion['choices'][0]['message']['content']


def chunk_on_pause(segments, n_pauses, min_words, max_words):
    """
    Segments continuous text into smaller chunks based on pauses 
    and word limits while retaining timing and content information.

    Setting n_pauses to -1 will result in chunks for every pause
    """
    pause = lambda segment1, segment2: segment2['start'] - segment1['end']

    # First Pass: Chunk text on all pauses & calculate time between
    times = []
    chunks = []
    text = ""
    text_start = 0

    for i in range(len(segments) - 1):
        time_between = pause(segments[i], segments[i+1])

        if time_between > 0 and text:
            chunks.append({"start":text_start, "end":segments[i]['end'], "time_between":time_between, "text":text})
            times.append(time_between)
            text = ""
            text_start = segments[i]['end']
        else:
            text += segments[i]['text']

    # Second Pass: Split chunks on the n longest pauses or max words & combine small chunks
    pause_cutoff = sorted(times, reverse=True)[n_pauses]

    output = []
    text = ""
    text_start = 0

    for c in chunks:
        text += c["text"]
        wordcount = len(text.split(' '))

        if wordcount >= min_words:
            if c["time_between"] > pause_cutoff or wordcount > max_words:
                output.append({"start":text_start, "end":c['end'], "wordcount":wordcount, "text":text})
                text = ""
                text_start = c['end']

    return output

def chapterize(chunks, verbose=True):
    sec_to_str =lambda t: str(0)+str(timedelta(seconds=int(t)))+',000'
    output = []

    q = "Create a single, short, and concise title for the following excerpt. Place ONLY the title between quotations " " in your response.\n"

    for i, c in enumerate(chunks):
        # Query Chapter
        result = query(q, c['text'])
        title = result[result.find('"'):result.rfind('"') + 1]
        time = f"[{sec_to_str(c['start'])} --> {sec_to_str(c['end'])}]"
        metadata = f"(Chunk {i+1}, {c['wordcount']} words, {str(timedelta(seconds=c['end']-c['start']))} mins)"

        # Chunk Metadata + Title
        output.append({"title":title, "time":time, "metadata":metadata})

        # Write Chunk
        if verbose:
            output[i]["text"] = c['text']
    return output
