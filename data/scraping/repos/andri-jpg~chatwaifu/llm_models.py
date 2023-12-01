import json
from pathlib import Path
from downloader import ModelDownloader
from llm_rs.langchain import RustformersLLM
from llm_rs import Bloom, SessionConfig, GenerationConfig, ContainerType, QuantizationType, Precision
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pathlib import Path

class ChainingModel:
    def __init__(self, model, name, assistant_name):
        self.model_download = ModelDownloader()
        with open('config.json') as self.configuration:
            self.user_config = json.load(self.configuration)
        with open('template.json') as self.prompt_template:
            self.user_template = json.load(self.prompt_template)
        meta = f"{model}.meta"
        model = f"{model}.bin"
        self.model = model
        if not Path(model).is_file():
            self.model_download.ask_download(f"https://huggingface.co/rustformers/redpajama-3b-ggml/resolve/main/RedPajama-INCITE-Chat-3B-v1-q5_1.bin", model)
        if not Path(meta).is_file():
            self.model_download.ask_download(f"https://huggingface.co/rustformers/redpajama-3b-ggml/resolve/main/RedPajama-INCITE-Chat-3B-v1-q5_1.meta", meta)

        self.name = name
        self.assistant_name = assistant_name
        self.names = f"<{name}>"
        self.assistant_names = f"<{assistant_name}>"
        
        self.stop_word = ['\n<human>:','<human>', '<bot>','\n<bot>:' ]
        self.stop_words = self.change_stop_words(self.stop_word, self.name, self.assistant_name)
        session_config = SessionConfig(
            threads=self.user_config['threads'],
            context_length=self.user_config['context_length'],
            prefer_mmap=False
        )

        generation_config = GenerationConfig(
            top_p=self.user_config['top_p'],
            top_k=self.user_config['top_k'],
            temperature=self.user_config['temperature'],
            max_new_tokens=self.user_config['max_new_tokens'],
            repetition_penalty=self.user_config['repetition_penalty'],
            stop_words=self.stop_words
        )

        template = self.user_template['template']

        self.template = self.change_names(template, self.assistant_name, self.name)
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "instruction"],
            template=self.template
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history")

        self.llm = RustformersLLM(
            model_path_or_repo_id=self.model,
            session_config=session_config,
            generation_config=generation_config,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

    @staticmethod
    def change_stop_words(stop_words, name, assistant_name):
        new_stop_words = []
        for word in stop_words:
            new_word = word.replace('bot', assistant_name)
            new_stop_words.append(new_word)
            new_word2 = word.replace('human', name)
            new_stop_words.append(new_word2)
        return new_stop_words

    @staticmethod
    def change_names(template, assistant_name, user_name):
        template = template.replace("bot", assistant_name)
        template = template.replace("human", user_name)
        return template

    def chain(self, input_text):
        prompt = self.prompt.generate_prompt({
            "chat_history": self.memory.export_memory(),
            "instruction": input_text
        })
        response = self.chain.generate(prompt)
        self.memory.add_message(input_text, "human")
        self.memory.add_message(response.choices[0].text.strip(), "ai")
        return response.choices[0].text.strip()


