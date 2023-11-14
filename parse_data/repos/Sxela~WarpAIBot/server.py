import os 
import shutil
from pathlib import Path
import torch

from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, GenerationConfig, TextStreamer, pipeline
import discord
from discord.ext import commands

import pytesseract
from PIL import Image


BOT_TOKEN = os.environ.get('BOT_TOKEN') #string
ADMIN = int(os.environ.get('ADMIN')) #int
CHANNELS = eval(os.environ.get('CHANNELS')) #list of ints
pytesseract.pytesseract.tesseract_cmd = os.environ.get('TESSERACT')

model_name_or_path = "TheBloke/WizardLM-13B-V1.2-GPTQ"
model_basename = "model"

max_body_len = 1800 #max length of answer+quoted docs
database_dir = './warpfusion_db/' #folder with txt files
max_history = 4 #remember last messages
chatgpt_max_history = 8
max_msg_len = 100 #when replying to user, limit doc quote to this length if a db doc has a discord url
max_nonmsg_len = 1000  #when replying to user, limit doc quote to this length for an internal db doc (no discord link)
min_comment_len = 4 #minimum length of a comment to be added to db
outdir = Path('./warpfusion_db/parsed') #parse output, needs to be db path or inside it. is purged before parsing

template = """
### Instruction: You're a WarpFusion script tech support agent who is talking to a customer. Make sure the customer has provided the WarpFusion script version, environment used, GPU specs, otherwise ask thme for it.
Use only the following information to answer in a helpful manner to the question. If you don't know the answer - say that you don't know.
Keep your replies short, compassionate, and informative.

'{context}'

{chat_history}
### Input: {question}
### Response: 
""".strip()
#LLM prompt

outdir.mkdir(exist_ok=True, parents=True)
device = torch.device('cuda')
use_triton = False
is_chatgpt = False
print(os.environ.keys())
if 'OPENAI_API_KEY' in os.environ.keys():
    print('Using chatgpt')
    is_chatgpt = True
    max_history = chatgpt_max_history
    from  langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model='gpt-3.5-turbo-16k')
else:
    from auto_gptq import AutoGPTQForCausalLM 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device='cuda:0',
            use_triton=use_triton,
            quantize_config=None)

    generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    streamer = TextStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, use_multiprocessing=False)

    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0.3,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True,
        generation_config=generation_config,
        streamer=streamer, batch_size=1)

    llm = HuggingFacePipeline(pipeline=pipe)

embeddings = HuggingFaceEmbeddings(
    model_name='embaas/sentence-transformers-multilingual-e5-base', model_kwargs={"device":device})

text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)

db = None
def update_db():
    global db
    old_len = len(db.get()['ids']) if db is not None else 0
    loader = DirectoryLoader(database_dir, glob="**/*txt")
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    try:
        db = Chroma.from_documents(texts, embeddings)
    except: pass
    db = Chroma.from_documents(texts, embeddings)
    new_len = len(db.get()['ids'])
    return new_len-old_len

old_len = update_db()
print(f'Added {old_len} new docs.')

prompt = PromptTemplate(input_variables=["context", "question","chat_history"], template=template)

memory_bank = {}

bot_prefix = "!" 
intents = discord.Intents.default() 
intents.message_content = True

# Create a bot instance
bot = commands.Bot(command_prefix=bot_prefix, intents=intents)

@bot.command()
async def fetch_all(ctx): 
    if outdir.exists:
        shutil.rmtree(outdir)
    if ctx.author.id != ADMIN: return
    await ctx.send("Parsing.")
    #parsing 
    channels = CHANNELS
    for channel in channels:
        await fetch_messages(ctx, channel_id=channel, limit=200)
    
    delta = update_db()
    await ctx.send(f"Finished. Added {delta} new documents.")
    return

@bot.command()
async def fetch_messages(ctx, channel_id: int, limit: int):
    if ctx.author.id != ADMIN: return
    # Get the channel object based on the provided channel_id
    channel = bot.get_channel(channel_id)

    chan_dir = outdir/f"{channel_id}_{channel.name}"
    chan_dir.mkdir(exist_ok=True, parents=True)

    if channel is None:
        await ctx.send("Channel not found.")
        return

    if isinstance(channel,discord.ForumChannel):
        #for a forum channel, get threads and make 1 doc per thread by combining all comments 
        print('got forum')
        print(len(channel.threads))
        # return
        threads = []
        async for thread in channel.archived_threads(limit=limit):
            threads.append(thread)
        for thread in channel.threads:
            threads.append(thread)
        for thread in threads:
            messages = []
            async for message in thread.history(limit=limit):
                messages.append(message)

            thread_dir = chan_dir/f"{thread.id}"
            thread_dir.mkdir(exist_ok=True, parents=True)
            text = [thread.name+'\n']

            write_messages_separately = False
            if len(messages)>0:
                for message in messages[::-1]:
                    if write_messages_separately:
                        # Get the message URL
                        message_url = f"https://discord.com/channels/{ctx.guild.id}/{thread.id}/{message.id}"
                        
                        # Process the messages as needed
                        outfile = thread_dir/f"{ctx.guild.id}_{thread.id}_{message.id}.txt"
                        if len(message.content.strip())>20:
                            with open(outfile, 'w', encoding="utf-8") as f:
                                f.write(thread.name+'\n'+message.content)
                    
                    else:
                        if len(message.content.strip())>min_comment_len:
                            text.append(f'{message.author.name}: {message.content}\n')
                
                outfile = thread_dir/f"{ctx.guild.id}_{thread.id}_{messages[0].id}.txt"
                with open(outfile, 'w', encoding="utf-8") as f:
                    f.writelines(text)

    else:
        #for a channel save each msg separately
        messages = []
        async for message in channel.history(limit=limit):
            messages.append(message)
        
        for message in messages:
            # Get the message URL
            message_url = f"https://discord.com/channels/{ctx.guild.id}/{channel_id}/{message.id}"
            
            # Process the messages as needed
            print(f"Author: {message.author.name}\nMessage Content: {message.content}\nMessage URL: {message_url}")
            outfile = chan_dir/f"{ctx.guild.id}_{channel_id}_{message.id}.txt"
            with open(outfile, 'w', encoding="utf-8") as f:
                f.write(message.content)


def custom_response(user_id, chat_id, server_id, user_message):
    user_key = user_id+chat_id+server_id
    print('user_key', user_key, user_id)
    if user_key not in memory_bank.keys():
        memory_bank[user_key] = ConversationBufferMemory(
                                        memory_key="chat_history",
                                        human_prefix="### Input",
                                        ai_prefix="### Response",
                                        input_key="question",
                                        output_key="output_text",
                                        return_messages=False
                                    )
    else: 
        #keep last 4 chat msgs 
        memory_bank[user_key].chat_memory.messages = memory_bank[user_key].chat_memory.messages[:max_history]
    
    chain = load_qa_chain(llm, chain_type="stuff", prompt = prompt, memory=memory_bank[user_key], verbose=False)
    question = user_message
    docs = db.similarity_search(question)
    with torch.autocast('cuda'), torch.inference_mode():
        answer = chain.run({"input_documents":docs, "question":question});

    # You can implement your custom logic here
    
    answer = answer.replace('`s', "'s").replace('`', '')
    response = answer

    lead = '\n\n**The following messages may be helpful:**\n\n'
    text = ''
    for doc in docs:
        
        url = doc.metadata['source']
        page_content = "\n".join([o for o in doc.page_content.splitlines() if o not in ['\n', '', ' ']]).replace('`s', "'s")
        if 'parsed' in url:
            text+=page_content[:max_msg_len]+'...\n'
            url = url.replace('\\','/').split('/')[-1][:-4].replace('_','/')
            url = f'https://discord.com/channels/{url}'
            text+=f'from: {url}\n\n'
        else: 
            text+=page_content[:max_nonmsg_len]+'...\n'
            text+=f'from: internal FAQ DB\n\n'
    lead+=text 
    response_tail = lead

    return response, response_tail

async def send_msgs(response, response_tail, message):
    user_id = message.author.id
    if len(response)+len(response_tail)>=max_body_len:
            await message.channel.send(f'<@{user_id}> {response[:max_body_len]}')
            await message.channel.send(f'<@{user_id}> {response_tail[:max_body_len]}')
    else:
            await message.channel.send(f'<@{user_id}> {response} {response_tail}')

config = '-l eng --oem 1 --psm 6'
# Function to extract text from an image using pytesseract
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, config=config)
    return text

# Event handler for when the bot is ready
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

# Event handler for when a message is received
@bot.event
async def on_message(message):
    # Avoid the bot responding to itself
    if message.author == bot.user:
        return

    # Check if the bot is mentioned in the message
    if bot.user.mentioned_in(message):
        async with message.channel.typing():
            user_id = message.author.id
            chat_id = message.channel.id
            server_id = message.guild.id
            user_message = message.content
            
            if len(message.attachments) > 0:
                for attachment in message.attachments:
                    if attachment.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                        # Download the image
                        img = f'image_{message.author.id}.jpg'
                        await attachment.save(img)

                        # Extract text from the image
                        ocr = extract_text_from_image(img)
                        
                        #delete the image
                        os.unlink(img)  
                        if len(ocr)>1800: ocr = ocr[-1800:]
                        await message.channel.send(f'<@{user_id}> I have recognized this in your image:\n```{ocr}```')
                        
                        if len(ocr)>500 and not is_chatgpt: ocr = ocr[-500:]
                        user_message += '\nI`m having this error message: \n' + ocr 

            # Call the abstract function and get the response
            response, response_tail = custom_response(user_id, chat_id, server_id, user_message)
            print('\nresponding with ', response)
            # Send the response as a mention
            await send_msgs(response, response_tail, message)

    await bot.process_commands(message)

# Run the bot
print('Launching the bot')
bot.run(BOT_TOKEN)
