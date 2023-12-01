import openai
import discord
import os
import subprocess
from discord.ext import commands
from VoiceList import voicelist
from rvc_infer import rvc_convert
from VoiceList import checarvoz

#  --------------------Discord-Token-------------------- #
DISCORD_TOKEN = open("Token.txt", "r").read()


#  ---------------------OpenAI-API---------------------- #
API_KEY = open("OA_API_KEY", "r").read()
openai.api_key = API_KEY

#  ---------------------Bot Set-up---------------------- #
intents = discord.Intents.default()
intents.message_content = True

# Initialize DEFVOICE to None
DEFVOICE = None

# Initialize the queue
queue = []

# if you are looking to change the PREFIX, it's
bot = commands.Bot(command_prefix=".", # <------ This line!
                   intents=intents)

# this will let us know if the bot is online
@bot.event
async def on_ready():
    print(f'{bot.user} esta listo!') 
    activity = discord.Game(name="Queues & yt-dlp")
    await bot.change_presence(activity=activity)

#  -----------------VOICE-SELECTION-------------------- #
@bot.command(help="Cambia la voz y el pitch default del bot.")
async def select(ctx, *, args):
    global DEFVOICE, DEFPITCH  # use the previous declared variables
    
    # argument split using comma: "voice, pitch"
    args_list = args.split(',', 1)

    # set the values for both variables
    DEFVOICE, DEFPITCH = map(str.strip, args_list)

    DEFVOICE = DEFVOICE.lower()
    # send "voice" to function in order to see if the voice is available.
    rvc_voice = checarvoz(voz=DEFVOICE)
    if rvc_voice is None:
    # if the function returns None, then it's not available.
        await ctx.send("Esa voz no esta disponible!")
        return

    # if you see this in discord, it means it works :)
    await ctx.send(f"voz cambiada a: {DEFVOICE}, pitch cambiado a: {DEFPITCH}")

#  ---------------------TTS/GPT------------------------ #
@bot.command(help="Has que el bot responda a tus preguntas.")
async def chat(ctx, *, args):
    # argument split using commas: "voice, pitch, user_response".
    voz, pitch, user_response = map(str.strip, args.split(',', 2))

    # this line of code is very important!! if you are looking to use TTS in English, 
    # you will have to replace "es-ES-AlvaroNeural" with an english speaker.
    # In order to see English voices, go ahead and run "edge-tts --list-voices".

    TTSVoice = "es-AR-TomasNeural" # <---------- This line here!!!!!


    # the function below will help us check if the voice we chose is available.
    rvc_voice = checarvoz(voz=voz)
    if rvc_voice is None:
    # if the function returns None, then it's not available.
        await ctx.send("Esa voz no esta disponible")
        return
    
    # Insert the user into the queue
    username = ctx.author.display_name
    queue.append(username)
    posicion = len(queue)

    # temp message
    tempmsg = await ctx.send("Generando respuesta...")

    # send "user_response" to ChatGPT.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": user_response}
        ]
    )
    # this will help us clean up ChatGPT's messy response.
    assistant_response = response['choices'][0]['message']['content']

    # assistant_response is going to be the new variable we use know
    # since it's ChatGPT's generated response.
    # So, we are going to send it to Edge-tts.
    # Note: Edge-tts will run locally within your computer.
    outputedgetts = f"chat_{posicion}.mp3"
    command = ["edge-tts", "--voice", TTSVoice, "--text", assistant_response, "--write-media", outputedgetts]
    subprocess.run(command)

    # sends "edgetts.mp3" to RVC.
    rvc_convert(model_path=rvc_voice,
                f0_up_key=pitch,
                input_path=outputedgetts,
                output_file_name=f"out{posicion}.wav"
                )
    
    # this will delete our temp message.
    await tempmsg.delete()

    # Let's assign the location of our RVC'd audio so then we can send it to discord.
    outputpath = f"output/out{posicion}.wav"
    audio_file = discord.File(outputpath)

    # send the file to discord
    await ctx.send(file=audio_file)

    # POP!
    queue.pop(0)

    # clean up!
    os.remove(outputedgetts)
    os.remove(outputpath)

#  ---------------------TTS----------------------------- #

@bot.command(help="Has que cualquier voz diga cualquier cosa!")
async def say(ctx, *, args):
    # argument split using commas: "voice, pitch, user_response".
    voice, pitch, user_response = map(str.strip, args.split(',', 2))

    # this line of code is very important!! if you are looking to use TTS in English, 
    # you will have to replace "es-ES-AlvaroNeural" with an english speaker.
    # In order to see English voices, go ahead and run "edge-tts --list-voices".

    TTSVoice = "es-AR-TomasNeural" # <----------- This line here!!!!!

    # send "voice" to function in order to see if the voice is available.
    rvc_voice = checarvoz(voz=voice)
    if rvc_voice is None:
    # if the function returns None, then it's not available.
        await ctx.send("Esa voz no esta disponible")
        return

    # TempMSG
    tempmsg = await ctx.send("Generando respuesta...")

    # Insert the user into the queue
    username = ctx.author.display_name
    queue.append(username)
    posicion = len(queue)

    # Send it to edge-tts.
    # Note: Edge-tts will run locally within your computer.
    outputedgetts = f"say_{posicion}.mp3"
    command = ["edge-tts", "--voice", TTSVoice, "--text", user_response, "--write-media", outputedgetts]
    subprocess.run(command)


    # Send it the generated audio to RCV.
    rvc_convert(model_path=rvc_voice,
                f0_up_key=pitch,
                input_path=outputedgetts,
                output_file_name=f"out{posicion}.wav"
                )
    
    # Delete TempMSG
    await tempmsg.delete()

    # Let's assign the location of our RVC'd audio so then we can send it to discord.
    outputpath = f"output/out{posicion}.wav"
    audio_file = discord.File(outputpath)

    # Send the audio file to discord
    await ctx.send(file=audio_file)

    # POP!
    queue.pop(0)

    # clean up!
    os.remove(outputedgetts)
    os.remove(outputpath)

#  ---------------------Copy audio---------------------- #

@bot.command(help="Has que cualquier voz cante o hable, funciona tambien con videos!")
async def audio(ctx):
    # These lines will check if the user input has any audio/video attachments.
    if len(ctx.message.attachments) == 0:
    # if none, end the process
        await ctx.send("Ocupo un archivo de audio/video!!")
        return
    
    # lets make sure the user has selected a voice, if not, lets instruct him how to do it.
    rvc_voice = checarvoz(voz=DEFVOICE)
    if rvc_voice is None:
        await ctx.send("Esa voz no esta disponible!")
        await ctx.send("Si quieres escoger una voz usa el siguiente comando: ```.select voz, pitch``` ")
        await ctx.send("Para ver la lista de voces usa: ```.voces```")
        return
      
    # Insert the user into the queue
    username = ctx.author.display_name
    queue.append(username)
    posicion = len(queue)

    # receives the attachment and saves it as "archivo" 
    archivo = ctx.message.attachments[0]


# this if statement will check if your input is a compatible audio file.
    if archivo.filename.endswith((".mp3", ".wav", ".flac")):
        # it will then name the input as "input.mp3"
        outputname = f"input{posicion}.mp3"
        tempmsg = await ctx.send(f"{posicion}, Generando audio...")
        
# this if statement will check if your input is a compatible video file.
    elif archivo.filename.endswith((".mp4", ".mov", ".mkv", ".webm")):
        # it will then name the input as "inputvideo.mp3".
        outputname = f"inputvideo{posicion}.mp3"
        tempmsg = await ctx.send(f"{posicion}, Generando video...")
        
        # using FFMPEG, it will then process the video so it gets converted as an mp3 file.
        command2 = ["ffmpeg", "-i", archivo.url, "-c:a", "aac", "-fs", "20M", outputname, "-y"]
        # Note: this will run locally in your computer.
        subprocess.run(command2)
        
# if your file is none of the above, then it's just not compatible, and ends the process.
    else:
        await ctx.send("Tu archivo no es compatible!")
        queue.pop(0)
        return


    # this will download the audio file that was provided by the user
    with open(outputname, "wb") as outputfile:
        outputfile.write(await archivo.read())

    # if, your input name is "inputvideo" it will get sent to 
    # FFMPEG in order to get the video without any audio.
    if outputname == f"inputvideo{posicion}.mp3":
        command3 = ["ffmpeg", "-i", outputname, "-an", f"input{posicion}.mp4", "-y"]
        subprocess.run(command3)
    
    # If you want to check the queue
    # print(queue) #<--- uncomment this
    # send the audio input to RVC
    rvc_convert(model_path=rvc_voice,
                f0_up_key=DEFPITCH,
                input_path=outputname,
                output_file_name=f"out{posicion}.wav"
                )
    
    #   delete tempMSG
    await tempmsg.delete()

    # Let's assign the location of our RVC'd audio so then we can use it on FFMPEG
    # or send it to discord.

    outputpath = f"output/out{posicion}.wav"

    audio_file = discord.File(outputpath)

    # if your outputname was: "inputvideo.mp3", it will get send yet again to FFMPEG
    # this time it will combine both the video with no audio and the RVC processed audio.
    if(outputname==f"inputvideo{posicion}.mp3"):
            # we need to close the file in order to delete it :-)
            audio_file.close()
            # FFMPEG command
            command4 = ["ffmpeg","-i", f"input{posicion}.mp4","-i", outputpath,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-fs", "20M", f"out{posicion}.mp4", "-y"]
            subprocess.run(command4)

            outputvid = f"out{posicion}.mp4"
            video_file = discord.File(outputvid)
            # send the file to discord
            await ctx.send(file=video_file)
            video_file.close()
            
            # Queue pop front
            queue.pop(0)

            # clean up!
            os.remove(outputvid)
            os.remove(f"input{posicion}.mp4")
            os.remove(f"inputvideo{posicion}.mp3")
            os.remove(outputpath)
    else:
        # if your audio was named "input{queue_position}.mp3", it will get sent over here!
        await ctx.send(file=audio_file)
        audio_file.close()

        # Queue pop front
        queue.pop(0)

        # clean up!
        os.remove(f"input{posicion}.mp3")
        os.remove(outputpath)
    
#  ----------------------YT-DLP audio---------------------- #
@bot.command(help="Lo mismo que audio, pero este descarga videos de youtube y otros sitios!")
async def url(ctx, user_response):
    # argument split using commas: "voice, pitch, user_response".
    # lets make sure the user has selected a voice, if not, lets instruct him how to do it.
    rvc_voice = checarvoz(voz=DEFVOICE)
    if rvc_voice is None:
        await ctx.send("Esa voz no esta disponible!")
        await ctx.send("Si quieres escoger una voz usa el siguiente comando: ```.select voz, pitch``` ")
        await ctx.send("Para ver la lista de voces usa: ```.voces```")
        return
    
    # Insert the user into the queue
    username = ctx.author.display_name
    queue.append(username)
    posicion = len(queue)

    # temp message
    tempmsg = await ctx.send(f"{posicion}, Descargando video...")
    # Setting up variables
    ytvid = f"output{posicion}.mp4"
    outputname = f"inputvideo{posicion}.mp3"

    # Youtube-dlp download
    commandyt = ["yt-dlp", "-f", "mp4",
             user_response, "-o", ytvid]
    subprocess.run(commandyt)
    await tempmsg.delete()

    tempmsg = await ctx.send(f"{posicion}, Generando video...")
    # using FFMPEG, it will then process the video so it gets converted as an mp3 file.
    commandff1 = ["ffmpeg", "-i", ytvid, "-map", "0:a:0", "-c:a", "mp3", "-fs", "20M", outputname, "-y"]
    # Note: this will run locally in your computer.
    subprocess.run(commandff1)

    # If you want to check the queue
    # print(queue) #<--- uncomment this
    # send the audio input to RVC
    rvc_convert(model_path=rvc_voice,
                f0_up_key=DEFPITCH,
                input_path=outputname,
                output_file_name=f"out{posicion}.wav"
                )
    #   delete tempMSG
    await tempmsg.delete()

    # Let's assign the location of our RVC'd audio so then we can use it on FFMPEG
    # or send it to discord.

    outputpath = f"output/out{posicion}.wav"
    audio_file = discord.File(outputpath)

    # if your outputname was: "inputvideo.mp3", it will get send yet again to FFMPEG
    # this time it will combine both the video with no audio and the RVC processed audio.
    audio_file.close()
    # FFMPEG command
    command4 = [
    "ffmpeg", "-i", ytvid, "-i", outputpath,
    "-c:v", "libx264", "-crf", "32",  # CRF/Custom Rate Factor
    "-vf", "scale=480:-2",  # Resolution
    "-map", "0:v:0", "-map", "1:a:0", "-fs", "20M", f"out{posicion}.mp4", # File size
    "-y"]

    subprocess.run(command4)

    outputvid = f"out{posicion}.mp4"
    video_file = discord.File(outputvid)
    # send the file to discord
    await ctx.send(file=video_file)
    video_file.close()
    
    # Remove from queue
    queue.pop(0)

    # clean up!
    os.remove(outputvid)
    os.remove(ytvid)
    os.remove(outputname)
    os.remove(outputpath)

# ----------------------Voice List---------------------- #
@bot.command(help="Lista de voces disponibles!")
async def voces(ctx):
    listadevoces = voicelist()
    await ctx.send("Esta es la lista de voces disponibles para el TTS! ")
    await ctx.send(listadevoces)
    await ctx.send("No olvides que el syntax para chat/say es: ```.commando voz, pitch, palabras```")

# ----------------------- End of code ------ Run ------ #
bot.run(DISCORD_TOKEN)
