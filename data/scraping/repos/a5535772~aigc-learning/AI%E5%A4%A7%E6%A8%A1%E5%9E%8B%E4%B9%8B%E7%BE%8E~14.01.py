import openai, os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

openai.api_key = os.environ.get("OPENAI_API_KEY")

llm = OpenAI(model_name="text-davinci-003", max_tokens=2048, temperature=0.5)

en_to_zh_prompt = PromptTemplate(
    template="请把下面这句话翻译成英文： \n\n {question}?", input_variables=["question"]
)

question_prompt = PromptTemplate(
    template="{english_question}", input_variables=["english_question"]
)

zh_to_cn_prompt = PromptTemplate(
    input_variables=["english_answer"],
    template="请把下面这一段翻译成中文： \n\n{english_answer}?",
)

question_translate_chain = LLMChain(llm=llm, prompt=en_to_zh_prompt, output_key="english_question")
qa_chain = LLMChain(llm=llm, prompt=question_prompt, output_key="english_answer")
answer_translate_chain = LLMChain(llm=llm, prompt=zh_to_cn_prompt)

from langchain.chains import SimpleSequentialChain

chinese_qa_chain = SimpleSequentialChain(
    chains=[question_translate_chain, qa_chain, answer_translate_chain], input_key="question",
    verbose=True)
answer = chinese_qa_chain.run(question="请你作为一个机器学习的专家，介绍一下CNN的原理。")
print(answer)
