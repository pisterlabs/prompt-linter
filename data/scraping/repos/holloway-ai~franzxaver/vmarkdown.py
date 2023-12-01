import time
import openai
import os
import logging
import json
import datetime
from pathlib import Path
import re

from app.core.config import settings
from app.transcriber.alignment import Transcript, MarkdownResponse, Aligner

logger = logging.getLogger(__name__)



system_prompt = """You are a transcription assistant.
Your goal is to format the given text so that it can be read while keeping the original text unchanged.
Try to execute it as accurately as you can.
"""
first_prompt_template = """
You are a transcription assistant. Process the given text into Markdown format fully, without any changes to the text. 
Put two carriage returns between paragraphs that make sense and are no longer than 100 tokens.
Group paragraphs into sections by inserting meaningful headings in Markdown format (line starting with ##). 
"""
second_prompt_template = """
You are a transcription assistant.
You'll be given a previous formatted fragment as context and the next portion of text to process as new text. 
Process only new text into Markdown format fully, without any changes to the text.
New text should follow context, do not change that order. Do not change the order of text.
Put two carriage returns between paragraphs that make sense and are no longer than 100 tokens.
Group paragraphs into sections by inserting meaningful headings in Markdown format (line starting with ##). 
"""


def get_prompt(text, context=None):
    if context is None:
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_prompt_template},
            {"role": "assistant", "content": "Sure, I'd be happy to! What's the text?"},
            {"role": "user", "content": text},
            {
                "role": "assistant",
                "content": "Ok, I get it. In the next message, I'll send you text in markdown format.",
            },
            {"role": "user", "content": "I'm ready to see the result"},
        ]
    else:
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_prompt_template},
            {
                "role": "assistant",
                "content": "Sure, I'd be happy to! What's the context?",
            },
            {"role": "user", "content": context},
            {"role": "assistant", "content": "I've got context. What's the new text?"},
            {"role": "user", "content": text},
            {
                "role": "assistant",
                "content": "Ok, I get it. In the next message, I'll send you text in markdown format.",
            },
            {"role": "user", "content": "I'm ready to see the result"},
        ]

    return prompt


def process(
    system_prompt, user_prompt, model="gpt-3.5-turbo", temperature=0.0, attempts=5
):
    retry_time = 0  
    def handle_error(e):
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        logger.warning("API error %s occurred.",e,exc_info=e)
        return retry_time
    
    for _ in range(attempts):
        try:

            if retry_time > 0:
                logger.warning("Retrying in %s seconds...",retry_time,exc_info=retry_time)
                time.sleep(retry_time)
                
            response = openai.ChatCompletion.create(
                model=model,
                messages=user_prompt,
                temperature=temperature,
                timeout=60,
                api_key=settings.OPENAI_API,
            )
            return response["choices"][0]["message"]["content"]
        
        except openai.error.Timeout as exception:
            retry_time = handle_error(exception)
        except openai.error.RateLimitError as exception:
            retry_time = handle_error(exception)
        except openai.error.APIError as exception:
            retry_time = handle_error(exception)
        except OSError as exception:
            retry_time = handle_error(exception)
        except openai.error.ServiceUnavailableError as exception:
            retry_time = handle_error(exception)
        
    raise Exception("Failed to get response")


def insert_slides(text, slides):
    if slides is None or len(slides) == 0 or len(text) == 0:
        return text
    re_block = re.compile(r"^\{~(?P<begin>\d+\.\d+)\}.*\{~(?P<end>\d+\.\d+)\} ?$")
    paragraphs = text.split("\n")
    current_slide = 0
    result = []
    for par in paragraphs:
        m = re_block.match(par)
        if m is not None:
            begin = float(m.group("begin"))
            end = float(m.group("end"))
            while current_slide < len(slides) and slides[current_slide][0] < end:
                result.append("")
                result.append(
                    f"![{{~{slides[current_slide][0]:.2f}}}]({slides[current_slide][1]})"
                )
                result.append("")
                current_slide += 1

        result.append(par)
    while current_slide < len(slides):
        result.append(
            f"![{{~{slides[current_slide][0]:.2f}}}]({slides[current_slide][1]})"
        )
        current_slide += 1

    return "\n".join(result)


def format_transcription(
    transcript_dict,
    progress_callback=None,
    step=0,
    total_steps=100,
    dump_path=None,
    slides=None,
):
    def dump_json(data, file_name):
        if dump_path is None:
            return
        dump_path.mkdir(parents=True, exist_ok=True)
        with (dump_path / file_name).open("w") as f:
            json.dump(data, f)

    def dump_text(text, file_name):
        if dump_path is None:
            return
        dump_path.mkdir(parents=True, exist_ok=True)
        with (dump_path / file_name).open("w") as f:
            f.write(text)

    dump_json(transcript_dict, "transcript.json")

    transcript = Transcript(transcript_dict)

    aligner = Aligner(
        transcript=transcript,
        block_size=1200,
        block_delta=400,
        context_size=400,
        context_size_delta=200,
    )

    stuck_count = 0

    progress_seg = (total_steps - step) / len(transcript)

    current_block = aligner.first_block()
    last_context_block_start = None
    while current_block:
 
        if last_context_block_start == aligner.context_block_start:
            logger.warning("Stuck at %s attempt %s ", aligner.context_block_start, stuck_count )
            stuck_count+=1
            if stuck_count > 3:
                break
        else:
            last_context_block_start = aligner.context_block_start
            stuck_count = 0
            
        state_prefix = f"{aligner.context_block_start}_{aligner.current_block_start}_{aligner.current_block_end}"
        dump_text(current_block, f"{state_prefix}_request.md")
        prompt = get_prompt(current_block)
        dump_json(prompt, f"{state_prefix}_prompt.json")

        temperature = 0.0
        formatted_result = process(
            system_prompt=system_prompt,
            user_prompt=prompt,
            model="gpt-3.5-turbo",
            temperature=temperature,
        )

        dump_text(formatted_result, f"{state_prefix}_result.md")
        formatted = MarkdownResponse(formatted_result)
        cut_pos = formatted.find_long_paragraph_start(400)
        if cut_pos > 0:
            if stuck_count < 3:
                cut_result = formatted.get_markdown_str(0, cut_pos)
                dump_text(cut_result, f"{state_prefix}_cut_{cut_pos}_result.md")
                formatted = MarkdownResponse(cut_result)
                logger.warning("cutting result to %s tokens",cut_pos,exc_info=cut_pos)
                temperature = 0.07
            else:
                logger.warning("Should be cutted at %s tokens but we skip as stuck",cut_pos,exc_info=cut_pos)
                

        else:
            temperature = 0.0
        current_block = aligner.push(formatted)

        if progress_callback is not None:
            end = current_block.find("\n")
            end = end if end > 0 else len(current_block)
            end = min(end, 20)
            progress_text = (
                current_block[:end].replace("\n", " ").replace("#", "").strip() + "..."
            )
            progress_callback(
                step=step + int(progress_seg * aligner.context_block_start),
                steps=total_steps,
                description="formatting: " + progress_text,
            )

    result = aligner.get_result()
    result = insert_slides(result, slides)
    dump_text(result, f"result.md")

    return result

