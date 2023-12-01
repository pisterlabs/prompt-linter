import threading
import time
from agent.factory import AgentFactory
from listeners.agent_listener import AgentListener
import speech_recognition as sr
import pyttsx3
from typing import Any, Dict, List, Sequence, Union


from langchain.callbacks import StdOutCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class VoiceListener(AgentListener):
    wake_word = "hey robot"
    ready_to_process = threading.Event()
    def __init__(self, factory: AgentFactory):
        super().__init__(factory)

    def listen(self, interrupt: threading.Event):
        self.ready_to_process.set()
        handler = SpeechCallbackHandler()
        while not interrupt.is_set():
            self.ready_to_process.wait()
            message = self.get_audio()
            if message.count(self.wake_word) > 0:
                self.ready_to_process.clear()
                message = message.replace(self.wake_word, "")
                agent = self.factory.new_k8s_engineer(handlers=[handler])
                agent({"input": message})
            self.ready_to_process.set()
        

    def start(self) -> threading.Event:
        # run listen in a separate thread
        interrupt = threading.Event()
        thread = threading.Thread(target=self.listen, args=(interrupt,))
        thread.start()
        return interrupt
    
    def get_audio(self):
        r = sr.Recognizer()
        r.energy_threshold = 4000
        with sr.Microphone(device_index=2) as source:
            audio = r.listen(source)
            said = ""

            try:
                said = r.recognize_whisper(audio)
                print(said)
            except Exception as e:
                print("Exception: " + str(e))
        # return lowercase, without commas
        return said.lower().replace(",", "")
    
class SpeechCallbackHandler(StdOutCallbackHandler):
    engine: Any
    voice_lock: threading.Lock

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.voice_lock = threading.Lock()
        try:
            self.engine = pyttsx3.init(driverName="nsss", debug=True)
            self.engine.setProperty('voice', 'com.apple.voice.compact.en-AU.Karen')
        except Exception as e:
            print(f"Error initializing pyttsx3: {e}")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        super().on_llm_start(serialized, prompts, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        super().on_llm_new_token(token, **kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        super().on_llm_end(response, **kwargs)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        super().on_llm_error(error, **kwargs)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        super().on_chain_start(serialized, inputs, **kwargs)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        super().on_chain_end(outputs, **kwargs)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        super().on_chain_error(error, **kwargs)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        super().on_tool_start(serialized, input_str, **kwargs)

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        super().on_tool_end(output, **kwargs)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        super().on_tool_error(error, **kwargs)

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        super().on_text(text, **kwargs)

    
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        super().on_agent_action(action, **kwargs)
        # get everything in action.log before "Action:"
        try:
            l = action.log.split("Action:")
            if len(l) < 2 or len(l[0]) == 0:
                return
            if self.engine._inLoop:
                self.engine.endLoop()
                time.sleep(0.5)
            # now get the part after "Thought:"
            # run the next part in a separate thread
            self.engine.say(l[0])
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in on_agent_action: {e}")
            

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent finish."""
        super().on_agent_finish(finish, **kwargs)
        try:
            if self.engine._inLoop:
                self.engine.endLoop()
                time.sleep(0.5)
            self.engine.say(finish.log)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in on_agent_finish: {e}")
