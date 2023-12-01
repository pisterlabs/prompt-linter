import textwrap
import unicodedata
import re
import time
import zlib
from typing import Iterator, TextIO, Union
import tqdm
import pyperclip
import urllib3
import openai
import tiktoken


def exact_div(x, y):
    assert x % y == 0
    return x // y


def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    return None if string == "None" else int(string)


def optional_float(string):
    return None if string == "None" else float(string)


def compression_ratio(text) -> float:
    return len(text) / len(zlib.compress(text.encode("utf-8")))


def format_timestamp(seconds: float, always_include_hours: bool = False, fractionalSeperator: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractionalSeperator}{milliseconds:03d}"


def write_txt(transcript: Iterator[dict], file: TextIO):
    for segment in transcript:
        print(segment['text'].strip(), file=file, flush=True)


def write_vtt(transcript: Iterator[dict], file: TextIO, 
              maxLineWidth=None, highlight_words: bool = False):
    iterator  = __subtitle_preprocessor_iterator(transcript, maxLineWidth, highlight_words)

    print("WEBVTT\n", file=file)

    for segment in iterator:
        text = segment['text'].replace('-->', '->')

        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

def write_srt(transcript: Iterator[dict], file: TextIO, 
              maxLineWidth=None, highlight_words: bool = False):
    """
    Write a transcript to a file in SRT format.
    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt
        result = transcribe(model, audio_path, temperature=temperature, **args)
        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    iterator  = __subtitle_preprocessor_iterator(transcript, maxLineWidth, highlight_words)

    for i, segment in enumerate(iterator, start=1):
        text = segment['text'].replace('-->', '->')

        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, fractionalSeperator=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, fractionalSeperator=',')}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

def __subtitle_preprocessor_iterator(transcript: Iterator[dict], maxLineWidth: int = None, highlight_words: bool = False): 
    for segment in transcript:
        words = segment.get('words', [])

        if len(words) == 0:
            # Yield the segment as-is or processed
            if maxLineWidth is None or maxLineWidth < 0:
                yield segment
            else:
                yield {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': process_text(segment['text'].strip(), maxLineWidth)
                }
            # We are done
            continue

        subtitle_start = segment['start']
        subtitle_end = segment['end']

        text_words = [ this_word["word"] for this_word in words ]
        subtitle_text = __join_words(text_words, maxLineWidth)
        
        # Iterate over the words in the segment
        if highlight_words:
            last = subtitle_start

            for i, this_word in enumerate(words):
                start = this_word['start']
                end = this_word['end']

                if last != start:
                    # Display the text up to this point
                    yield {
                        'start': last,
                        'end': start,
                        'text': subtitle_text
                    }
                
                # Display the text with the current word highlighted
                yield {
                    'start': start,
                    'end': end,
                    'text': __join_words(
                        [
                            {
                                "word": re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", word)
                                        if j == i
                                        else word,
                                # The HTML tags <u> and </u> are not displayed, 
                                # # so they should not be counted in the word length
                                "length": len(word)
                            } for j, word in enumerate(text_words)
                        ], maxLineWidth)
                }
                last = end

            if last != subtitle_end:
                # Display the last part of the text
                yield {
                    'start': last,
                    'end': subtitle_end,
                    'text': subtitle_text
                }

        # Just return the subtitle text
        else:
            yield {
                'start': subtitle_start,
                'end': subtitle_end,
                'text': subtitle_text
            }

def __join_words(words: Iterator[Union[str, dict]], maxLineWidth: int = None):
    if maxLineWidth is None or maxLineWidth < 0:
        return " ".join(words)
    
    lines = []
    current_line = ""
    current_length = 0

    for entry in words:
        # Either accept a string or a dict with a 'word' and 'length' field
        if isinstance(entry, dict):
            word = entry['word']
            word_length = entry['length']
        else:
            word = entry
            word_length = len(word)

        if current_length > 0 and current_length + word_length > maxLineWidth:
            lines.append(current_line)
            current_line = ""
            current_length = 0
        
        current_length += word_length
        # The word will be prefixed with a space by Whisper, so we don't need to add one here
        current_line += word

    if len(current_line) > 0:
        lines.append(current_line)

    return "\n".join(lines)

def process_text(text: str, maxLineWidth=None):
    if (maxLineWidth is None or maxLineWidth < 0):
        return text

    lines = textwrap.wrap(text, width=maxLineWidth, tabsize=4)
    return '\n'.join(lines)

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def download_file(url: str, destination: str):
        with urllib3.request.urlopen(url) as source, open(destination, "wb") as output:
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

# -------------used for text post processing tab----------------
system_prompt = "You are a helpful assistant."
user_prompt = "请帮我把下面的文本纠正错别字并添加合适的标点符号，返回的消息只要处理后的文本："

def get_chunks(s, maxlength, separator=None):
    start = 0
    end = 0
    while start + maxlength  < len(s) and end != -1:
        if separator is not None:
            end = s.rfind(separator, start, start + maxlength + 1)
            segment = s[start:end]
            yield segment.replace(separator, "")
            start = end +1
        else:
            end = start + maxlength
            yield s[start:end]
            start = end

    yield s[start:]
    
def post_processing(text, use_chatgpt, user_token, apply_correction, auto_punc, separator, remove_words):
    # print(f"==>> separator: {separator}")
    original_separator1 = " "
    original_separator2 = ","
    
    if use_chatgpt == True:
        if user_token == "":
            text = "请先设置你的OpenAI API Key，然后再重试"
            return text
        else:
            text = chat_with_gpt(text, system_prompt, user_prompt)
            return text
    # 对于长文本需要先分段再推理，推理完再合并    
    elif auto_punc == True:
        # 自动分段文本之前先去除原有的标点符号
        text = text.replace(original_separator1, "") 
        text = text.replace(original_separator2, "") 
        import paddlehub as hub
        model = hub.Module(name='auto_punc', version='1.0.0')
        t3 = time.time()
        # split long text to short text less than max_length and store them in list
        max_length = 256
        chunks = list(get_chunks(text, max_length))
        results = []
        results = model.add_puncs(chunks, max_length=max_length)
        text = "，".join(results) # 分段处硬编码成使用中文逗号分割
        t4 = time.time()
        print("Auto punc finished. Cost time: {:.2f}s".format(t4-t3))
    # print(f"==>> text after auto punc: {text}")
    else:    
        # 将空格全部统一替换成一种分隔符
        if separator == "\\n":
            # 直接使用separator会无法换行
            text = text.replace(original_separator1, "\n") 
            text = text.replace(original_separator2, "\n") 
        elif separator != "": # 当separator为空时不替换，便于该tab可以单独使用
            text = text.replace(original_separator1, separator)
            text = text.replace(original_separator2, separator)

    if apply_correction == True:
        import pycorrector
        print("Start correcting...")
        t1 = time.time()
        text, detail = pycorrector.correct(text)
        t2 = time.time()
        print("Correcting finished. Cost time: {:.2f}s".format(t2-t1))
        print(f"==>> detail: {detail}")

    # 去掉语气词
    t5 = time.time()
    remove_words = remove_words.split("，") + remove_words.split(",") + remove_words.split(" ")
    for word in remove_words:
        text = text.replace(word, "")
    t6 = time.time()
    print("Remove words finished. Cost time: {:.2f}s".format(t6-t5))

    return text
    
def replace(text, src_word, target_word):
    text = text.replace(src_word, target_word)
    return text
    
def copy_text(text):
    pyperclip.copy(text)

def num_tokens_from_messages(message):
    """Return the number of tokens used by a list of messages."""
    model="gpt-3.5-turbo-0613"
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(message, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(message, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    num_tokens += tokens_per_message
    message = user_prompt + message
    num_tokens += len(encoding.encode(message))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return f"Tokens 计数: {num_tokens}"

def on_token_change(user_token):
    openai.api_key = user_token

def chat_with_gpt(input_message, system_prompt, user_prompt, temperature=0, max_tokens=4096):
    system_content = [{ "role": "system", "content": system_prompt }]
    user_content = [{ "role": "user", "content":  user_prompt + input_message }]
    try:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=system_content + user_content, temperature=temperature, max_tokens=max_tokens)
        response_msg = completion.choices[0].message['content']

        prompt_tokens = completion['usage']['prompt_tokens']
        completion_tokens = completion['usage']['completion_tokens']
        total_tokens = completion['usage']['total_tokens']
        print(f"==>> prompt_tokens: {prompt_tokens}")
        print(f"==>> completion_tokens: {completion_tokens}")
        print(f"==>> total_tokens: {total_tokens}")
        return response_msg
    
    except Exception as e:
        return f"Error: {e}"

    
# -------------used for text post processing tab----------------