import base64
import os
import re
import sys
from dotenv import load_dotenv
import numpy as np
import openai
from asgiref.sync import sync_to_async
from src.langchain_agent import ChatAgent
from src.whispers_engine import process_audio_data

# Load the environment variables
load_dotenv()


class WhispersSession:

    def __init__(self, transcriptionQueue, reasoningQueue, socketManager):
        self.transcriptionQueue = transcriptionQueue
        self.reasoningQueue = reasoningQueue
        self.socketManager = socketManager
        self.wakeWords = ["Hey Froggy", "Hey froggy", "Hey Froggy",
                          "Hey froggy", "Hey, Froggy", "Hey, froggy",
                          "Hey Froggie", "Hey, Froggie", "Hey Froggie",
                          "Hey froggie", "Hey, froggie", "Hey, froggie",
                          "Hey froggie", "Froggy", "Froggie", "froggie", "froggy"]
        self.froggySession = False
        self.previousRecording = np.array([])
        self.froggyMessage = ''
        self.froggySessionCounter = 0

    def __base64_to_narray(self, convertible):
        # Convert the string to a bytes object
        convertible = base64.b64decode(convertible)

        # Convert the bytes object to a NumPy array
        convertible = np.frombuffer(convertible, np.int16).astype(
            np.float32)

        return convertible

    def __clean_text(self, message: str, wakeWord: str):
        """
        Remove wake word from message and return capitalized text.
        """
        message = re.findall(f"!?({wakeWord}.\s|{wakeWord}\s)(.+)", message)
        print('message:', message)

        if message:
            message = message[0][1].strip().capitalize()
        else:
            message = ""
        return message

    async def process_audio_data_from_queue(self, websocket):
        print("Froggy Session:", self.froggySession)
        print("Froggy Session Counter:", self.froggySessionCounter)
        # Get audio data from the queue
        currentRecording = await self.transcriptionQueue.get()

        # Process the audio data and broadcast the transcription result
        currentRecording = self.__base64_to_narray(currentRecording)
        previous_recording_length = self.previousRecording.shape[0]
        concatenatedRecording = np.concatenate([
            self.previousRecording[-previous_recording_length//3:], currentRecording])
        self.previousRecording = currentRecording
        # print('concatenatedRecording:', concatenatedRecording)
        transcription = await process_audio_data(concatenatedRecording, prompt="Hey Froggy,") if self.froggyMessage == '' else await process_audio_data(concatenatedRecording, prompt=self.froggyMessage)

        if self.froggySession is False and len(self.froggyMessage) > 1:
            data = {
                "status": "broadcasting",
                "transcription": self.froggyMessage,
            }
            await self.socketManager.broadcast(websocket, data)
            await self.reasoningQueue.put(self.froggyMessage)
            self.froggyMessage = ""

        if transcription:

            # appends cleaned text if wake word is present in message
            for wakeWord in self.wakeWords:
                if wakeWord in transcription:
                    self.froggySession = True
                    self.froggyMessage += self.__clean_text(
                        transcription, wakeWord)

                    # only one wake word is allowed per message
                    break

        if (transcription is None or transcription is (" " or "")) and self.froggySessionCounter >= 5:
            self.froggySession = False
            self.froggySessionCounter = 0

        if self.froggySession == True and (transcription != None or transcription != (" " or "")):
            self.froggySessionCounter += 1
            self.froggyMessage += transcription.rstrip().lower()
            data = {
                "status": "broadcasting",
                "transcription": self.froggyMessage,
            }
            await self.socketManager.broadcast(websocket, data)

        self.transcriptionQueue.task_done()

    # async def get_ai_response(self, websocket):
    #     # Set the API key for OpenAI
    #     openai.api_key = os.getenv("OPENAI_API_KEY")
    #     # Get the prompt for the OpenAI model
    #     prompt = os.getenv("PROMPT_PREFIX")

    #     # Get the transcription from the queue
    #     transcription = await self.reasoningQueue.get()

    #     # Use the OpenAI API to get a response for the transcription
    #     response = await sync_to_async(openai.Completion.create)(
    #         model="text-davinci-003",
    #         prompt=f"{prompt}Q: {transcription}\nA: ",
    #         temperature=0,
    #         max_tokens=100,
    #         top_p=1,
    #         frequency_penalty=0,
    #         presence_penalty=0,
    #     )

    #     if response:
    #         print("response:", response)
    #         data = {
    #             "status": "broadcasting",
    #             "reasoning": response.choices[0].text.strip(),
    #         }
    #         await self.socketManager.broadcast(websocket, data)

    #     self.reasoningQueue.task_done()

    async def get_ai_response(self, websocket):
        # Get the transcription from the queue
        transcription = await self.reasoningQueue.get()
        history_array = []

        print("\n\n#### INPUT ####\n")
        print(transcription)
        print("\n\n#### INPUT ####\n")

        chat_agent = await sync_to_async(ChatAgent)(history_array=history_array)

        try:
            reply = chat_agent.agent_executor.run(input=transcription)

        except ValueError as inst:
            print('ValueError:\n')
            print(inst)
            reply = "Sorry, there was an error processing your request."

        print("\n\n#### REPLY ####\n")
        print(reply)
        print("\n\n#### REPLY ####\n")

        pattern = r'\(([a-z]{2}-[A-Z]{2})\)'
        # Search for the local pattern in the string
        match = re.search(pattern, reply)

        language = 'en-US'  # defaut
        if match:
            # Get the language code
            language = match.group(1)

            # Remove the language code from the reply
            reply = re.sub(pattern, '', reply)

        print("LANG: ", language)

        sys.stdout.flush()
        if reply:
            data = {
                "status": "broadcasting",
                "reasoning": reply.strip(),
            }
            await self.socketManager.broadcast(websocket, data)
        self.reasoningQueue.task_done()
