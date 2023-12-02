from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

rephrase_template = """
You must follow the steps described below:

Step 1:
Напиши, как можно изменить некоторые слова в вопросе, чтобы он трактовался однозначно. Обрати внимание, что некоторые слова можно воспринимать по–разному. Избавься от такой неоднозначности.
Покажи только исправленный вопрос.

Вопрос: {input}

Покажи только новый вопрос
"""

rephrase_prompt = PromptTemplate(
    template=rephrase_template,
    input_variables=["input"]
)

llm = ChatOpenAI(temperature=0.5, verbose=True, model_name="gpt-3.5-turbo")

rephrase_chain = LLMChain(llm=llm, prompt=rephrase_prompt, verbose=True)


def rephrase(input: str) -> str:
    return rephrase_chain.run(
        input=input,
    )
