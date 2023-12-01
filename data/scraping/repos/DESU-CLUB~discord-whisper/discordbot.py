import discord
from discord.ext import commands
from discord.sinks import WaveSink
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from whisper_jax import FlaxWhisperPipline
import tempfile
import jax.numpy as jnp
import time
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.llms.fake import FakeListLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
import tiktoken
from langchain.docstore.document import Document
intents = discord.Intents.all()
intents.members = True

#llm = OpenAI(temperature=0)
llm = FakeListLLM(responses = ['Able to run', "it ha been a log time,"])
model =FlaxWhisperPipline("openai/whisper-large-v2", dtype=jnp.float16)
bot = commands.Bot(command_prefix='!', intents=intents)
executor = ThreadPoolExecutor(max_workers=5)
content_string = '''\
You are SummarizeGPT, a LLM that summarizes long lengths of discord dialogues into their main ideas.
Summarize the following dialogue: 
{text}

TLDR:'''
content_template = PromptTemplate(input_variables=['text'], template=content_string)


#Tiktoken function to count no of tokens
def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = tokenizer.encode(text)
    return len(tokens)

template_len = count_tokens(content_string)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8000 - template_len,
    chunk_overlap=100,
    length_function=count_tokens,
    separators=["\n\n","\n",'']
)

                           

class MyHelp(commands.HelpCommand):
    async def send_bot_help(self, mapping):
        embed = discord.Embed(title="Help")
        for cog, commands in mapping.items():
           command_signatures = [self.get_command_signature(c) for c in commands]
           if command_signatures:
                cog_name = getattr(cog, "qualified_name", "No Category")
                embed.add_field(name=cog_name, value="\n".join(command_signatures), inline=False)

        channel = self.get_destination()
        await channel.send(embed=embed)


    async def send_command_help(self, command):
        embed = discord.Embed(title=self.get_command_signature(command), color=discord.Color.random())
        if command.help:
            embed.description = command.help
        if alias := command.aliases:
            embed.add_field(name="Aliases", value=", ".join(alias), inline=False)

        channel = self.get_destination()
        await channel.send(embed=embed)

bot.help_command = MyHelp()
   
@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

async def start_recording(ctx,sink,my_callback,*args):
    ctx.voice_client.start_recording(sink,my_callback,ctx,*args)
    return

def transcriber(audio):
    start = time.time()
    text = model(audio)
    print('Finish transcription', time.time()-start)
    os.unlink(audio)
    return text['text']

async def my_callback(sink, *args):
        # Save the recorded audio to a file
        ctx,summarize = args
        print(summarize)
        if summarize == 'True':
            summarize = True
        loop = asyncio.get_event_loop()
        start = time.time()
        progress = await ctx.send('Reading bytes...')
        data = await loop.run_in_executor(executor,read_file,sink)
        #run whisper on here
        print('Reading bytes',time.time()-start)
        await progress.edit('Transcribing...')
        start = time.time()
        text = await loop.run_in_executor(executor,transcriber,data)
        print('Transcribed', time.time()-start)
        #If more than 4000 words, split into chunks
        if summarize == True:
            await progress.edit('Summarizing...')
            start = time.time()
            text = await summarize_text(text)
            print('Summarized', time.time()-start)
        if len(text) > 4000:
            for x in range(0, len(text), 4000):
                if x == 0:
                    await progress.edit(f'{text[x:x+4000]}')
                await ctx.send(f'{text[x:x+4000]}')
        else:
            await progress.edit(text)


def read_file(sink):
    try:
        all_audio = sink.get_all_audio()
        if not all_audio:
            print("No audio data in sink")
            return None

        # Get WAV data bytes
        wav_data_bytes = all_audio[0].getvalue()

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_filename = temp_file.name

        # Write the WAV data bytes to the temporary file
        with open(temp_filename, 'wb') as f:
            f.write(wav_data_bytes)

        # Delete the temporary file
        return temp_filename

    except Exception as e:
        print(f'Error in read_file: {e}')
        return None

##Count tokens, then summarize
async def summarize_text(text):
    texts = text_splitter.split_text(text)
    docs = [Document(page_content= text) for text in texts]
    print(docs, docs[0])
    chain = load_summarize_chain(llm, chain_type='stuff', prompt=content_template)
    results = await chain.arun(docs)
    print('Summarization ran')
    print(results)
    return ''.join(results) if len(docs) == 1 else '\n\n'.join(results)

async def async_generate(chain,doc):
    print(doc.page_content)
    resp = await chain.arun(doc)
    return resp


@bot.command(help = "Record audio in a voice channel\nSummarize flag: Default=> False,\n True=> Summarize the text")
async def transcribe(ctx,summarize = False):
        #delete file        
    if ctx.author.voice is None:
        await ctx.send("You are not in a voice channel!")
        return
    elif ctx.voice_client != None and ctx.author.voice.channel != ctx.voice_client.channel:
        await ctx.send("I'm already in another voice channel!")
        return
    else:
        channel = ctx.author.voice.channel
        if ctx.voice_client == None:
            await channel.connect()

    if ctx.voice_client == None:
        await ctx.send('I am not in a voice channel!')
        return
    
    elif ctx.voice_client and ctx.voice_client.is_connected():
        print('Working!')
        sink = WaveSink()
        await start_recording(ctx,sink,my_callback,summarize)
        return
    else:
        await ctx.send('Unknown Error Found!')
        return

@bot.command(help = "Stop recording audio in a voice channel, and transcribes it")
async def stop(ctx):
    if ctx.voice_client is None:
        await ctx.send("I am not in a voice channel!")
        return

    try:
        # Check if the bot is actively recording before stopping the recording
        ctx.voice_client.stop_recording()
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")
    
    
@bot.command(help = "Disconnect the bot from the voice channel")
async def disconnect(ctx):
    if ctx.voice_client is None:
        await ctx.send("I am not in a voice channel!")
        return
    await ctx.voice_client.disconnect()



if __name__ == "__main__":
    with open('mybot.txt','r') as f:
        key = f.read().strip()
        bot.run(key)


    
