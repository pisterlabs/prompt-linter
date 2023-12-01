import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from podcastOutroWriterPlugins.baseOutroWriterPlugin import BaseOutroWriterPlugin


class OutroWriterPlugin(BaseOutroWriterPlugin):
    def identify(self) -> str:
        return "🎸 outro writer plugin"

    def writeOutro(self, stories, introText):
        print("Writing funny Outro")
        llm = OpenAI(
            model=os.getenv("OPENAI_MODEL_SUMMARY"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS_OUTRO")),
            temperature=0.3,
        )
        templateString = os.getenv("OUTRO_TEMPLATE_STRING")
        prompt = PromptTemplate(
            input_variables=["introText"],
            template=templateString,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(introText)


plugin = OutroWriterPlugin()
