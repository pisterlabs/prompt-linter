from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

translation_template = (
    "Translate the following text from {source_language} into {target_language}: {text}"
)
translation_prompt = PromptTemplate(
    input_variables=["source_language", "target_language", "text"],
    template=translation_template,
)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

source_language = "English"
target_language = "German"
text = "What a wonderful day!"

translated_text = translation_chain.predict(
    source_language=source_language, target_language=target_language, text=text
)
print(translated_text)