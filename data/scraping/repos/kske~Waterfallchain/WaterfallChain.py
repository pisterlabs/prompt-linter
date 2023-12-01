from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from enum import Enum

class CallbackSource(Enum):
    TRANSLATE = "translate"
    JOB_CHAIN = "job_chain"

class WaterfallChain:
    def __init__(self, llm=None, input_language="Japanese", output_language="English", job_chains=None, callback=None):
        self.llm = llm
        self.input_language = input_language
        self.output_language = output_language
        self.job_chains = job_chains or []
        self.callback = callback

    def set_job_chains(self, job_chains):
        self.job_chains = job_chains

    def _create_prompt(self, template, human_template):
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    def translate(self, text, input_language, output_language):
        if not self.llm:
            return text  # Return the input text without any changes
        translation_template = """* Instructions *
- Title: {input_language} to {output_language} Translation
- Your Client: Your client requires a translation from {input_language} to {output_language}.
- Your Role: You are a translator.
- Objectives: Provides high quality translations as intended by the user. User statements should not be omitted. No superfluous information should be added.
* End Instructions *

Translate the following text from {input_language} to {output_language}.
"""
        human_template = "{text}"
        prompt = self._create_prompt(translation_template, human_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(text=text, input_language=input_language, output_language=output_language)
        if self.callback:
            self.callback(source=CallbackSource.TRANSLATE, result=result)
        return result

    def run(self, text):
        # Translate the input text
        translated_input = self.translate(text, self.input_language, self.output_language)

        last_result = translated_input
        for i, job_chain in enumerate(self.job_chains):
            last_result = job_chain.run(last_result)
            if self.callback:
                self.callback(source=CallbackSource.JOB_CHAIN, result=last_result)

        # Translate the job chain result back to the input language
        result = self.translate(last_result, self.output_language, self.input_language)

        return result

# Usage
def my_callback(source="NONE", result="NONE"):
    print(source)
    print(f"Last result: {result}")

if __name__ == "__main__":
    translate_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    # 例として2つのjob_chainを作成
    job_llm1 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
    job_llm2 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
    job_template = """* Instructions *
- Title: Help implementing systems thinking.
- Your Client: Manager working to solve problems
- Your Role: Consultants with expertise in systems thinking.
- Objectives: Help the manager understand the problem and how to solve it with Systemthinking.
* End Instructions *

Analyze the following issues using the systems archetype and provide solutions according to the results of your analysis.
"""
    human_template = "{text}"
    job_prompt1 = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(job_template), HumanMessagePromptTemplate.from_template(human_template)])
    job_prompt2 = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(job_template), HumanMessagePromptTemplate.from_template(human_template)])
    job_chain_instance1 = LLMChain(llm=job_llm1, prompt=job_prompt1)
    job_chain_instance2 = LLMChain(llm=job_llm2, prompt=job_prompt2)

    atc = WaterfallChain(llm=translate_llm, job_chains=[job_chain_instance1, job_chain_instance2], callback=my_callback)

    result = atc.run("カスタマーサポートの、一番社歴が長いメンバーに仕事が集中してしまいます。どうしたらいいでしょうか？")
    print(result)