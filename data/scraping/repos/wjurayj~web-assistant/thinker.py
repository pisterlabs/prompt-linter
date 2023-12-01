import openai
import asyncio
import logging
import time
import textwrap
import threading
from utils import Message
# from tools.notepad import Notepad
from tools.toggler import Toggler
from tools.toolkit import ToolKit
from tools.notepad import NotePad

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filename=f'logs/thinker_{time.time()}.log',
    filemode='w'
)

class Thinker:
    def __init__(self, config, logger=None):
        self.name = config.get("name", "Chatter")
        self.model = config.get("model", "gpt-3.5-turbo-0301")
        self.speaker = None
        self.logger = logger if logger else logging.getLogger()
        self.logger.warning(f"Initialized Thinker using model {self.model}")

        prime = Message(**{
            "role": "system",
            "content": f"You are {self.name}, a friendly and helpful coding assistant."
        })
        
        init = [
            Message(**{
                "role": "user",
                "content": f"You are being used with a visually impaired text to speech accessory that uses a headset for interaction with you. Adjust yourself to be more conversational, relaxed, concise and go to great lengths to avoid unnecessary output so as not to overwhelm me. Never mention being a language model AI, policies or similar. Try to keep responses short unless I say to expand upon it. If you understand reply “ready” without further explanation."
            }),
            Message(**{
                "role": "assistant",
                "content": "ready"
            })
        ]

        self.prime = [prime]
        
        # self.context = []
        
        # Track what's actually been said, from user's perspective
        self.utterances = init
        
        self.toolkit = ToolKit(self)
        self.toolkit.add_tool(NotePad())
        
        self.tools = [Toggler(self)] #, Notepad()]
        
    def inject(self, message, meta={}):
        self.utterances.append(Message(message, meta=meta))

    def check_tools(self, message):
        self.toolkit.check_tools(self.utterances)
        # This function iterates over self.tools (or self.toolkit.tools)
        # Builds a list of questions for GPT-3
        # Packs into one prompt and feeds them in
        # Parses outputs using dict of indices (lookup each tool to get its prompt indices in choices)
        print(f'previous model is {self.model}')
        for t in self.tools:
            t.process(message)
        print(f'current model is {self.model}')
        pass

    def receive(self, message):
        self.utterances.append(Message(**{
            "role": "user",
            "content": message
        }))
        self.logger.info(f"> {message}")
        # you could run your secondary task thread(s?) here...
        # > each should ...
        self.check_tools(message)
        # thread = threading.Thread(target=self.check_tools, args=(message,), daemon=True)
        # thread.start()
        return None

    def verbalize(self):
        pass
    
    def process(self, nfails = 4, webapp=False):
        for i in range(nfails):
            try:
                return self._process(webapp)
            except openai.error.RateLimitError:
                self.logger.info(f"Hit rate limit on try #{i}")
                time.sleep(2**i)
            except openai.error.APIConnectionError:
                self.logger.info(f"API connection error on try #{i}")
                time.sleep(2**i)
        self.logger.info(f"Unable to process input after {i} tries")

    def _process(self, webapp=False):

        #sned request for stream
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[m.to_openai() for m in self.prime + self.utterances],
            temperature=1,
            # max_tokens=256,
            top_p=0.98,
            frequency_penalty=0.5,
            presence_penalty=0.2,
            stream=True
        )
        
        utterance = ""
        buffer = ""
        for chunk in response:
            ch = chunk['choices'][0]['delta'].get('content', '')
            buffer += ch
            #needs to always yield, and have the chatter wrapper print if necessary
            # if webapp:
            yield ch
            # else:
            #     print(ch, end="", flush=True)
            if '\n' in ch: #break down by more frequent/all punctuation, like a period?
                #send to speaker by paragraph, etc.
                if self.speaker:
                    # print('verbalizing!')
                    
                    #this blocks text rendering a little, which I don't want
                    self.speaker.verbalize(buffer)#, eg
                utterance += buffer
                buffer = ""
                pass
        #handle the last paragraph
        if buffer:
            if self.speaker: #don't write if we're in a code block; instead the UI should render this
                # print('verbalizing the last bit!')
                self.speaker.verbalize(buffer)
            utterance += buffer
        print()

        self.utterances.append(Message(**{
            "role": "assistant",
            "content": utterance
        }))
        self.logger.info(f"% {utterance}")