import os
import speech_recognition as sr
import nextcord
from nextcord.ext import commands
import openai
from pydub import AudioSegment
from io import BytesIO


def create_pcm_stream(packets):
    for packet in packets:
        yield b"".join(packet)


class VoiceChannelRecorder:
    def __init__(self, voice_client):
        self.packets = []
        self.buffer = []

        self.voice_client = voice_client
        self.voice_client.listen(self)

    def recv_packet(self, packet):
        self.buffer.append(packet)

    def start(self):
        self.voice_client.pause()

    def stop(self):
        self.voice_client.stop_listening(self)
        self.voice_client.resume()
        self.packets.append(self.buffer)
        self.buffer = []


class ChatBot(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def ask(self, ctx):
        if ctx.author.voice:
            # Connect to the voice channel and start recording
            voice_channel = ctx.author.voice.channel
            voice_client = await voice_channel.connect()

            recorder = VoiceChannelRecorder(voice_client)
            recorder.start()

            # Prompt the user to ask a question
            message = await ctx.send("Please ask your question.")

            # Wait for the user to stop speaking
            await nextcord.utils.sleep_until(10)  # 10 seconds wait, customize as needed

            # Stop recording and disconnect
            recorder.stop()
            await voice_client.disconnect()

            # Show a loading message while the chatbot is processing the question
            loading_message = await ctx.send("Processing your question...")

            # Transcribe the audio
            audio = b"".join(p for p in create_pcm_stream(recorder.packets))
            audio_file = BytesIO(audio)
            audio_segment = AudioSegment.from_file(audio_file, format="wav")
            audio_file = BytesIO()
            audio_segment.export(audio_file, format="wav")
            audio_file.seek(0)

            # Use OpenAI API to transcribe the user's question
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

            # Generate a response using OpenAI's API
            prompt = f"Q: {transcript}\nA:"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.7,
            )

            # Send the response to the text channel
            await loading_message.delete()
            await message.edit(content=response.choices[0].text)
        else:
            await ctx.send("Please join a voice channel to ask a question.")


def setup(bot):
    bot.add_cog(ChatBot(bot))