import discord, os, requests, asyncio
from discord import app_commands
import openai

discord_token = os.environ["DISCORD_TOKEN_BOT1"]
models_path = os.environ["AIMODELSPATH"]

guildObject = discord.Object(id = 1084203391519051786)

class aclient(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.synced = False
    
    async def on_ready(self):
        await self.wait_until_ready()
        if not self.synced:
            await tree.sync(guild = guildObject)
            self.synced = True

        print(f"We have logged in as {self.user}")

client = aclient()
tree = app_commands.CommandTree(client)

# Default values
pygmalionState = False
pygmalionCharacter = None

@tree.command(name = "test", description = "Used for testing purposes", guild = guildObject)
async def self(interaction: discord.Interaction):
    await interaction.response.defer()
    await interaction.followup.send(content="Testing", ephemeral=True, silent=True)

@tree.command(name = "deletelastmessages", description = "Delete the x last messages", guild = guildObject)
async def self(interaction: discord.Interaction, count:int = 1):
    await interaction.response.defer()
    messages_list = []
    
    async for message in interaction.channel.history(limit=count+1):
        if message.author.id != client.user.id:
            await message.delete()
    interaction.followup.send(content=f"Deleted last {count} messages.", ephemeral=True, silent=True)

@tree.command(name = "clear", description = "Clear the current channel", guild = guildObject)
async def self(interaction: discord.Interaction):
    await interaction.response.defer()
    if interaction.user.id == 152917625700089856:
        await interaction.channel.purge()
    else:
        interaction.followup.send(content="You do not have permissions to do this!", delete_after=5, ephemeral=True, silent=True)

'''Custom'''
from logic import *

'''AI general'''
device = 'cuda'
import torch

'''Stable-diffusion | Image generation'''
from diffusers import StableDiffusionPipeline
from diffusers import logging as difflog

from transformers import logging as translog

translog.set_verbosity_error()
difflog.set_verbosity_error()

'''OpenJourney'''
@tree.command(name = "openjhelp", description="Get help with OpenJourney", guild = guildObject)
async def self(interaction: discord.Interaction):
    await interaction.response.defer()
    message = "**Prompt:** the text that the model with generate an image with\n"
    message += "**Negative prompt:** the text that the model with avoid generating an image with\n"
    message += "**Count:** amount of images to generate\n"
    message += "**Seed:** the seed to use when generating the image\n"
    message += "**Guidance scale:** how similar the image is to the prompt\n"
    message += "**Steps:** More steps = more quality and time to generate\n"
    message += "**Width and height:** image dimensions\n"
    await interaction.followup.send(message, ephemeral=True, silent=True)

@tree.command(name = "openjourney", description="Generate text to image using OpenJourney", guild = guildObject)
async def self(interaction: discord.Interaction, prompt:str, negative_prompt:str = None, guidance_scale:float = 7.5, count:int = 1, seed:int = None, steps:int = 50, width:int = 512, height:int = 512):
    await interaction.response.defer()
    
    if width % 8 != 0 and height % 8 != 0:
        await interaction.followup.send(content=f"Height and Width have to be divisible by 8 but are {height} and {width}\nFor height subtract: {height % 8} (change to {height-(height % 8)})\nFor width subtract: {width % 8} (change to {width-(width % 8)})", silent=True, ephemeral=True)
        return

    if count < 1 or count > 10:
        await interaction.followup.send(content="I cannot generate less than 1 or more than 10 pictures!", ephemeral=True, silent=True)
        return
    
    if not prompt:
        await interaction.followup.send(content="No prompt given", ephemeral=True, silent=True)

    files = []

    for i in range(count):
        print(f"Generating image {i+1} of {count}")
        try:
            model_id = models_path+"/openjourney"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
            generator = torch.Generator(device)

            generator = generator.manual_seed(seed+i) if seed else generator.manual_seed(generator.seed()+i)
            
            outputtext = f"**Text prompt:** {prompt}\n"
            outputtext += f"**Negative text prompt:** {negative_prompt}\n"
            outputtext += f"**Count:** {count}\n"
            outputtext += f"**Seed:**  {generator.initial_seed()}\n"
            outputtext += f"**Guidance scale:** {guidance_scale}\n"
            outputtext += f"**Steps:** {steps}\n"
            outputtext += f"**Size:** {width}x{height}\n"

            filename = f"{generator.initial_seed()}_{guidance_scale}-{steps}.png"

            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                width=width,
                height=height,
                generator=generator
            )
            
            for im, image in enumerate(result.images):
                # If NSFW Detected
                if result.nsfw_content_detected[im] == True:
                    outputtext += f"NSFW detected on image {im + 1} of {count}\n"

                name = f"{im+1}_{filename}"
                image.save(name, 'PNG')
                files.append(discord.File(fp=name, description=f"Prompt: {prompt}\nNegative prompt: {negative_prompt}"))
        except RuntimeError as e:
            if 'out of CUDA out of memory' in str(e):
                outputtext += f"Out of memory: try another prompt"

    await interaction.followup.send(content=outputtext, files=files, silent=True)

    for file in files:
        os.remove(file.filename)

@tree.command(name = "openjourneywithincrease", description="Generate text to image using OpenJourney", guild = guildObject)
async def self(interaction: discord.Interaction, prompt:str, negative_prompt:str = None, increase_guidance_by:float = 2.0, guidance_start:float = 0.0, count:int = 1, seed:int = None, steps:int = 50, width:int = 512, height:int = 512, creategif:bool = False):
    await interaction.response.defer()

    if width % 8 != 0 and height % 8 != 0:
        await interaction.followup.send(content=f"Height and Width have to be divisible by 8 but are {height} and {width}\nFor height subtract: {height % 8} (change to {height-(height % 8)})\nFor width subtract: {width % 8} (change to {width-(width % 8)})", silent=True, ephemeral=True)
        return

    if increase_guidance_by <= 0.0 or increase_guidance_by > 10.0:
        await interaction.followup.send(content="Guidance should not be lower or equal to zero. And it should not be higher than 10", ephemeral=True, silent=True)
        return
    
    if count > 10:
        await interaction.followup.send(content="Discord can only send the first 10 images!", ephemeral=True, silent=True)

    if count < 1:
        await interaction.followup.send(content="I cannot generate less than 1 image!", ephemeral=True, silent=True)
        return
    
    if not prompt:
        await interaction.followup.send(content="No prompt given", ephemeral=True, silent=True)
        return

    files = []
    files2 = []

    guidance_scale_list = []
    for generation in range(count):
        guidance_scale_list.append(increase_guidance_by*generation+guidance_start)

    generator = torch.Generator(device)
    if not seed:
        seed = generator.seed()

    outputtext = f"**Text prompt:** {prompt}\n"
    outputtext += f"**Negative text prompt:** {negative_prompt}\n"
    outputtext += f"**Seed:**  {seed}\n"
    outputtext += f"**Guidance scale start:** {guidance_start}\n"
    outputtext += f"**Guidance scale increase:** {increase_guidance_by}\n"
    outputtext += f"**Count:** {count}\n"
    outputtext += f"**Steps:** {steps}\n"
    outputtext += f"**Size:** {width}x{height}\n"

    for i, guidance_scale in enumerate(guidance_scale_list):
        try:
            generator = generator.manual_seed(seed)

            model_id = models_path+"/openjourney"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
            
            filename = f"{generator.initial_seed()}_{guidance_scale}-{steps}.png"
            iter_filename = f"imagestoGIF/{i+1}_{filename}"
            print(f"Generating: {filename}")

            result = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                width=width,
                height=height,
                generator=generator
            )
            
            for im, image in enumerate(result.images):
                # If NSFW Detected
                if result.nsfw_content_detected[im] == True:
                    outputtext += f"NSFW detected on image {i + 1} of {count}\n"

                image.save(iter_filename, 'PNG')
                if not (len(files) >= 10):
                    files.append(discord.File(fp=iter_filename, description=f"Prompt: {prompt}\nNegative prompt: {negative_prompt}"))
                else:
                    files2.append(iter_filename)
        except RuntimeError as e:
            if 'out of CUDA out of memory' in str(e):
                outputtext += f"Out of memory: try another prompt"
                break
        except discord.app_commands.errors.CommandInvokeError as e:
            print(e)
            break

    if files:
        await interaction.followup.send(content=outputtext, files=files, silent=True)
    # If error occured
    else:
        await interaction.followup.send(content=outputtext, silent=True, ephemeral=True)

    if creategif and files:
        output_gif = create_gif(find_image_paths(r'imagestoGIF'), r'imagestoGIF/output.gif', 1)
        gif_file = discord.File(fp=output_gif, description=f"GIF of output images")
        files.append(gif_file)

        await interaction.followup.send(content='Created GIF from images', file=gif_file, silent=True)

    for file in files:
        print(f"Deleting: imagestoGIF/{file.filename}")
        os.remove(f"imagestoGIF/{file.filename}")

    for filename in files2:
        print(f"Deleting: {filename}")
        os.remove(f"{filename}")

from diffusers import StableDiffusionImg2ImgPipeline

@tree.command(name = "openjourneyimg", description="Generate image to image using OpenJourney", guild = guildObject)
async def self(interaction: discord.Interaction, file: discord.Attachment, prompt:str = "", negative_prompt:str = None, seed:int = None, guidance_scale:float = 7.5, steps:int = 50):
    await interaction.response.defer()

    if not file or not file.filename.endswith(('.png', '.jpg', '.webp', 'jpeg')):
        await interaction.followup.send(content="Invalid file extension", ephemeral=True, silent=True)
        return
    
    r = requests.get(file.url)
    with open(file.filename, 'wb') as f:
        f.write(r.content)
    
    filename = file.filename

    files = []
    files.append(discord.File(fp=filename, description="Prompt file"))

    try:
        model_id = models_path+"/openjourney"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        generator = torch.Generator(device)
        if not seed:
            generator.seed()

        outputtext = f"**Text prompt:** {prompt}\n"
        outputtext += f"**Negative text prompt:** {negative_prompt}\n"
        outputtext += f"**Seed:**  {generator.initial_seed()}\n"
        outputtext += f"**Guidance scale:** {guidance_scale}\n"
        outputtext += f"**Steps:** {steps}\n"

        if seed:
            generator = generator.manual_seed(seed)

        with Image.open(filename) as im:
            result = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                image=im,
                generator=generator
            )
        
        for i, image in enumerate(result.images):
            # If NSFW Detected
            if result.nsfw_content_detected[i] == True:
                outputtext += f"NSFW Detected on image\n"

            name = f"{i+1}_{filename}"
            image.save(name, 'PNG')
            files.append(discord.File(fp=name, description=f"Image of {prompt}"))
    except RuntimeError as e:
        if 'out of CUDA out of memory' in str(e):
            outputtext += f"Out of memory: try another prompt"

    await interaction.followup.send(content=outputtext, files=files, silent=True)

    for file in files:
        os.remove(file.filename)

'''Whisper | Transcription of audio and video'''
import whisper

import validators
from pytube import YouTube

# Attach audio file and output text
@tree.command(name = "whisper", description="Generate transcriptions and detect language using OpenAI's Whisper model", guild = guildObject)
async def self(interaction: discord.Interaction, file:discord.Attachment = None, url:str = None, transcribe:bool = True, prompt:str = "", detect:bool = False):
    await interaction.response.defer()
    
    if not transcribe and not detect:
        await interaction.followup.send(content="No operation given; use transcribe and/or detect!", ephemeral=True, silent=True)
        return
    
    if not file and not url:
        await interaction.followup.send(content="No file or url attached", ephemeral=True, silent=True)
    
    if file and url:
        await interaction.followup.send(content="You can only add a file __or__ an url!", ephemeral=True, silent=True)

    if file:
        if not file.filename.endswith(('.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm')):
            await interaction.followup.send(content="Invalid file extension", ephemeral=True, silent=True)
            return
        
        filename = file.filename
        print(f"Downloading {filename}")
        r = requests.get(file.url)
        with open(filename, 'wb') as f:
            f.write(r.content)

    elif url:
        if (validators.url(url)):
            output = f"<{url}>\n"
            filename = "yt_download.mp3"
            YouTube(url).streams.filter(only_audio=True).first().download(filename=filename)
            if not filename:
                await interaction.followup.send(content="Could not find file, contact admin", ephemeral=True, silent=True)
                return
            print(f"Downloaded {filename}")
        else:
            await interaction.followup.send(content="Invalid url!", ephemeral=True, silent=True)
            return

    model_name = "medium"
    model = whisper.load_model(model_name, device=device)

    if detect:
        audio = whisper.load_audio(filename)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        output += f"Detected language: {max(probs, key=probs.get)}"
    if transcribe:
        result = model.transcribe(filename, initial_prompt=prompt) if prompt else model.transcribe(filename)
        if detect:
            output += " | "
        output += f"Transcribed {filename}"

    if result['text']:
        inputPrompt = discord.File(fp=filename)
        with open(f"transcription_{filename}.txt", "w") as f:
            f.write(result['text'].strip())
        outputfile = discord.File(fp=f"transcription_{filename}.txt")

        files = [inputPrompt, outputfile]
        await interaction.followup.send(content=output, files=files, silent=True)
    else:
        await interaction.followup.send(content="Could not create a transcription", ephemeral=True, silent=True)

    for file in files:
        os.remove(file.filename)

'''Clip | Guessing'''
import clip
import numpy as np

# Attach image and output text
@tree.command(name = "clip", description="Attach an image and possible guesses to make AI guess what is in image", guild = guildObject)
async def self(interaction: discord.Interaction, file:discord.Attachment, prompt:str):
    await interaction.response.defer()

    if not file:
        await interaction.followup.send(content="No file attached!", ephemeral=True, silent=True)
        return
    
    if not file.filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        await interaction.followup.send(content="Invalid file extension", ephemeral=True, silent=True)
        return
    
    if not prompt:
        await interaction.followup.send(content="No prompt given!", ephemeral=True, silent=True)
        return

    print(f"Downloading {file.filename}")
    r = requests.get(file.url)
    with open(file.filename, 'wb') as f:
        f.write(r.content)
    
    filename = file.filename

    model_name = "ViT-B/32"

    # Load model
    model, preprocess = clip.load(model_name, device=device)

    image = preprocess(Image.open(filename)).unsqueeze(0).to(device)
    possibilities = prompt.split(", ")
    textprob = clip.tokenize(possibilities).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(textprob)
        logits_per_image, logits_per_text = model(image, textprob)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # Create list of percentages
    probas = []
    for item in probs[0]:
        probas.append(float(np.format_float_positional(item, precision=4)))
    
    # Create and sort list with possibilites and percentages combined
    list_sorted = sorted(zip(possibilities, probas), key=lambda x: x[1], reverse=True)
    print(list_sorted)

    # Format list
    text_list = []
    for item in list_sorted:
        text_list.append(f"{item[0]}: {item[1] * 100:.2f}%")
    output = "\n".join(text_list)
    print(text_list)
    
    # Send output to discord
    if output:
        imagePrompt = discord.File(fp=filename)

        with open(f"guess_{filename}.txt", "w") as f:
            f.write(output)
        outputfile = discord.File(fp=f"guess_{filename}.txt")

        files = [imagePrompt, outputfile]
        await interaction.followup.send(content="", files=files, silent=True)
        
        # Remove files
        for file in files:
            os.remove(file.filename)

@tree.command(name = "chatgpt", description="Chat Completion with gpt-3.5-turbo", guild = guildObject)
async def self(interaction: discord.Interaction, prompt:str):
    if prompt == 'Test':
        openai.api_key = os.environ["OPENAI_API_KEY"]
        responses = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        )

        print(responses['choices'][0]['message']['content'])

from transformers import AutoTokenizer, AutoModelForCausalLM

@tree.command(name = "pygmalion", description="Conversational bot set state with character", guild = guildObject)
async def self(interaction: discord.Interaction, state:bool, character_name:str = 'Discord user', character_description:str = 'The average discord user, likes to be edgy, cringe and sometimes very offensive.'):
    await interaction.response.defer()
    if interaction.channel.id != 1091464570943574076:
        channel = client.get_channel(1091464570943574076)
        interaction.followup.send(content=f"Due to it being a conversational bot, this bot is only available in the {channel} channel.", ephemeral=True, silent=True)
        return
    
    global pygmalionState, pygmalionCharacter, pygmalionCharacterName
    pygmalionState = state
    pygmalionCharacter = character_description
    pygmalionCharacterName = character_name
    if not pygmalionState:
        status = "Pygmalion is now disabled"
    else:
        status = f"{pygmalionCharacterName} is now **active** with the persona: {pygmalionCharacter}"
    print(status)
    await interaction.followup.send(status, silent=True)
    # await asyncio.sleep(5)

    if pygmalionState and pygmalionCharacterName and pygmalionCharacter:
        global pygmaTokenizer, pygmaModel
        model_id = models_path+"/pygmalion-350m"
        pygmaTokenizer = AutoTokenizer.from_pretrained(model_id)
        print('Set tokenizer')
        pygmaModel = AutoModelForCausalLM.from_pretrained(model_id).to(torch.device("cuda:0"))
        print('Set model (ready)')

@client.event
async def on_message(ctx: discord.Message):
    if ctx.author.id == client.user.id:
        return
    
    if ctx.channel.id != 1091464570943574076:
        return

    if not pygmalionState:
        await ctx.channel.send(content="Pygmalion is not active!", delete_after=3, silent=True)
        return

    if not pygmalionCharacter:
        await ctx.channel.send(content="No character to play!", delete_after=3, silent=True)
        return

    try:
        messages_list = []
        async for message in ctx.channel.history(limit=200):
            messages_list.append(f"{message.author.name}: {message.content}\n")

        input_message = messages_list.pop()
        messages = "".join(reversed(messages_list))

        input_text = f"{pygmalionCharacterName}'s Persona: {pygmalionCharacter}\n<START>\n"
        input_text += messages
        input_text += f"{input_message}\n"
        input_text += f"{pygmalionCharacterName}: "
        print(input_text)

        input_ids = pygmaTokenizer.encode(input_text, return_tensors='pt').to('cuda')
        print("test")
        
        output = pygmaModel.generate(input_ids, max_length=1000, do_sample=True, temperature=0.7).to(torch.device("cuda:0"))
        output_text = pygmaTokenizer.decode(output[0], skip_special_tokens=True)

        await ctx.channel.send(output_text, silent=True)
        print(output_text)
    except:
        pass

client.run(discord_token)