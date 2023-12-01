import os
import time
from openai import OpenAI as OpenAI_
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
import sys
if os.path.dirname(os.path.realpath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import keys
from langchain.chains import LLMChain
import os
import FlightOffer.apiDataHandler as apiDataHandler
import keys

class AIregular:
    def __init__(self, openAI_APIKEY):
        self.openAI_APIKEY = openAI_APIKEY
        os.environ['OPENAI_API_KEY'] = openAI_APIKEY
        self.chat_model = ChatOpenAI()

    def returnAnswer(self, userInput):
        prompt = "{message}"

        chat_prompt = ChatPromptTemplate.from_messages([prompt])

        chain = LLMChain(
        llm=ChatOpenAI(temperature="0.7", model_name='gpt-3.5-turbo'),
        prompt=chat_prompt,
        verbose=True
        )
        answer = chain.run({"message": userInput})

        return answer
    
    def returnDocAnswer(self, userInput, textFiles=[], imageFiles=[]):
        client = OpenAI_()

        for index, file_ in enumerate(textFiles):
            textFiles[index] = client.files.create(
            file=file_,
            purpose='assistants'
            ).id

        fileAssistant = client.beta.assistants.create(
        instructions="You are a helpful robot.",
        model="gpt-4-1106-preview",
        tools=[{"type": "retrieval"}],
        file_ids=[]
        )

        if len(imageFiles) > 0:
            output_vision = self.processImages(userInput, imageFiles)
            if len(textFiles) <= 0:
                return output_vision
            
        if len(textFiles) > 0:
            content_text = "Answer the following prompt based on text documents attached to this message.\nPrompt: " + userInput
            if len(imageFiles) > 0:
                content_text += "\n\nAditional important information from attached files you dont have direct access to, which you should take into consideration when coming up with an answer:\n" + output_vision
        else:
            content_text = userInput
            if len(imageFiles) > 0:
                content_text += "\n\nAditional important information from attached files you dont have direct access to, which you should take into consideration when coming up with an answer:\n" + output_vision

        thread = client.beta.threads.create(
        messages=[
            {
            "role": "user",
            "content": content_text,
            "file_ids": textFiles
            }
        ]
        )

        assistant_id=fileAssistant.id

        run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id
        )

        while True:
            time.sleep(3)
            run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
            )
            print(run)
            print(run.status)

            if run.status == "failed":
                return "There was an error extracting data."
            if run.status == "completed":
                break
        
        print("Done")

        messages = client.beta.threads.messages.list(
        thread_id=thread.id
        )
        print("Answer:\n", messages.data[0].content[0].text.value)
        answer = messages.data[0].content[0].text.value
        apiDataHandler.delete_assistant(fileAssistant.id, keys.openAI_APIKEY)

        for file_ in textFiles:
            apiDataHandler.delete_file(file_, keys.openAI_APIKEY)

        return answer
    
    def processImages(self, userInput, imageFiles):
        client = OpenAI_()

        messages = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": userInput},
            ],
            }
        ]

        for imageFile in imageFiles:
            img = {
                "type": "image_url",
                "image_url": {
                    "url": imageFile,
                },
            }
            messages[0]["content"].append(img)

        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=500,
        )

        print(response.choices[0].message.content)
        return response.choices[0].message.content

#AIregular_ = AIregular(keys.openAI_APIKEY)
#AIregular_.processImages("what is in these images?", ["https://i.gyazo.com/52237d1ce662e613af2950c3a351a6c8.jpg", "https://i.gyazo.com/98ca775b51faa2a19fa6d7fbf588e5e3.png"])
#with open("Nabavno.pdf", 'rb') as file:
#    AIregular_.returnDocAnswer(userInput="Summarize everything that is attached to this message.", textFiles=[file], imageFiles=["https://i.gyazo.com/52237d1ce662e613af2950c3a351a6c8.jpg", "https://i.gyazo.com/98ca775b51faa2a19fa6d7fbf588e5e3.png"])