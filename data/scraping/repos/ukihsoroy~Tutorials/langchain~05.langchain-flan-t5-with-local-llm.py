from langchain import PromptTemplate, HuggingFaceHub, LLMChain

from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

model_id = 'google/flan-t5-large'# go for a smaller model if you dont have the VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)

template = """Question: {question}
        Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=local_llm)

question = "What is the capital of France?"

print(llm_chain.run(question))