import asyncio
import subprocess
from tempfile import TemporaryDirectory
import anthropic
import os
from deepgram import Deepgram
from assembly_transcription import stream_transcripts
from claude import stream_claude_sentences
from speech_synthesis import generate_speech_from_text
from microphone_stream import get_microphone_stream

# TODO:
# Higher quality Gordon.
# No weird asterix stuff.

SYSTEM_PROMPT = f""" You are a voice only assistant so can only output speech and not actions/emojis etc.
Never output bullet points.
Never output things in asterixes (e.g. *laughs*). Avoid actions alltogether.
You should never output the following charachter: *
You should never output "*" in any form.
I want you to act as a nice, cheeky and fun AI assistant. You should act like my friend and answer concisely like we are chatting.
It should be a back and worth conversation with you occasionally asking me questions.
{anthropic.AI_PROMPT} Sure! I am happy to do that and will make sure to only output dialougue and ASCII charachters and never output the "*" character. I'll make my answers short and snappy!
{anthropic.HUMAN_PROMPT} And only say one or two sentences at a time!
{anthropic.AI_PROMPT} I'll keep my answers snappy.
{anthropic.HUMAN_PROMPT} To stress, I really don't want you answering with lots of sentences. Just one or two.
{anthropic.AI_PROMPT} You don't have to tell me twice."""

#SYSTEM_PROMPT = f""" I want you to role play as Gordon Ramsay. Be funny and entertaining, but answer concisely as we are having a back and forth conversation.
#You are a voice only assistant so don't write anything that can't be converted to speech (like emojis or *starts clapping* etc). {anthropic.AI_PROMPT} I am happy to do an impression of Gordon Ramsay! What do you want to waste my time with today!?{anthropic.HUMAN_PROMPT} """
            

async def generate_text(text_to_synthesise_queue: asyncio.Queue):
    """
    Go all the way through to generating Claude responses.
    Then add the text to a queue so we can do the TTS in parallel.
    """

    anthropic_client = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])
    
    total_conversation = f"{anthropic.HUMAN_PROMPT} {SYSTEM_PROMPT}"
    
    # Make sure we don't record microphone when bot is speaking.
    is_safe_to_record = asyncio.Event()
    is_safe_to_record.set()

    with get_microphone_stream(is_safe_to_record) as microphone_stream:
        async for transcript in stream_transcripts(microphone_stream):
        #for transcript in ["Tell me three jokes."]:
            print(f"Transcript: {transcript}")

            total_conversation += f" {transcript} {anthropic.AI_PROMPT}"

            # Don't record any more speech once Claude starts writing.
            is_safe_to_record.clear()
            
            sentences_seen_in_convo_turn = 0
            current_chunk = ""

            async for claude_response_sentence in stream_claude_sentences(anthropic_client, total_conversation):
                print(f"Claude: {claude_response_sentence} ")
                total_conversation += f" {claude_response_sentence}"
                sentences_seen_in_convo_turn += 1
                
                # Start generating on the first two sentence ASAP.
                if sentences_seen_in_convo_turn <= 2:
                    await text_to_synthesise_queue.put(claude_response_sentence)
                    continue
                    
                # For later sentences, we can do them is batches to make the speech more natural.
                current_chunk += f" {claude_response_sentence}"
                if sentences_seen_in_convo_turn % 2 == 0:
                    await text_to_synthesise_queue.put(current_chunk)
                    current_chunk = ""
                    
            # Start generating leftover sentences.
            if current_chunk != "":
                await text_to_synthesise_queue.put(current_chunk)

            total_conversation += anthropic.HUMAN_PROMPT
                
            # Fine to record again once we have played all the audio
            # in the current conversation turn.
            # Unsure of a better way of doing this.
            convo_turn_sentinel = asyncio.Event()
            await text_to_synthesise_queue.put(convo_turn_sentinel)
            await convo_turn_sentinel.wait()
            is_safe_to_record.set()
                

async def generate_all_speech(text_to_synthesise_queue, speech_to_play_futures_queue):
    """
    This gets somewhat complicated. We want to boost latency by generating the speech in any order.
    However, the speech must be played in order. I miss trio...
    """
    
    # ElevenLabs only gives us 2 concurrents :(
    semaphore = asyncio.Semaphore(2)

    async def generate_with_future(text, future, semaphore):
        async with semaphore:
            audio = await generate_speech_from_text(text)
        future.set_result(audio)

    async with asyncio.TaskGroup() as tg:           
        while True:
            text_to_synthesise = await text_to_synthesise_queue.get()
            
            # Propagate the sentinel to the audio player.
            if isinstance(text_to_synthesise, asyncio.Event):
                convo_turn_sentinel = text_to_synthesise
                await speech_to_play_futures_queue.put(convo_turn_sentinel)
                continue

            future = asyncio.Future()
            await speech_to_play_futures_queue.put(future)
            tg.create_task(generate_with_future(text_to_synthesise, future, semaphore))


async def play_audio(speech_to_play_futures_queue):
    # I think this sometimes borks for headphones?
    
    mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
    mpv_process = subprocess.Popen(
        mpv_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    while True:
        speech_to_play_future = await speech_to_play_futures_queue.get()

        # Once we set this sentinel, we are good to start recording audio again.
        if isinstance(speech_to_play_future, asyncio.Event):
            convo_turn_sentinel = speech_to_play_future
            convo_turn_sentinel.set()
            continue

        audio = await speech_to_play_future

        mpv_command = ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"]
        mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        mpv_process.stdin.write(audio)
        # Add some checkpoints as the audio playing blocks.
        # This can prevent submitting TTS requests.
        await asyncio.sleep(0.01)
        mpv_process.stdin.close()
        await asyncio.sleep(0.01)
        mpv_process.wait()
        await asyncio.sleep(0.01)
        

async def main():

    async with asyncio.TaskGroup() as tg:
        text_to_synthesise_queue = asyncio.Queue()
        speech_to_play_futures_queue = asyncio.Queue()
        tg.create_task(generate_text(text_to_synthesise_queue))
        tg.create_task(generate_all_speech(text_to_synthesise_queue, speech_to_play_futures_queue))
        tg.create_task(play_audio(speech_to_play_futures_queue))
    

if __name__ == "__main__":
    asyncio.run(main())
