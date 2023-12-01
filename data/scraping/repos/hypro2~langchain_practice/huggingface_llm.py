from langchain import HuggingFacePipeline
from langchain import PromptTemplate
from langchain.schema import BaseOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, TextStreamer


def llama_prompt():
    template = """
<s>[INST] <<SYS>>
Act as a Machine Learning engineer who is teaching high school students.
<</SYS>>

{text} [/INST]
"""

    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )
    return prompt


def huggingface_llm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 # torch_dtype=torch.float16,
                                                 # trust_remote_code=True,
                                                 # device_map="auto"
                                                 )


    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = 128
    generation_config.temperature = 0.0001
    generation_config.top_p = 0.95
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15
    generation_config.streamer=TextStreamer(tokenizer)

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )

    llm = HuggingFacePipeline(model_id = model_name, pipeline=text_pipeline, model_kwargs={"temperature": 0})

    return llm

### TEST ###

class CustomSpaceSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(" ")

if __name__=="__main__":
    # MODEL_NAME = "TheBloke/Llama-2-7b-Chat-GPTQ"
    model_name = "facebook/opt-125m"
    llm = huggingface_llm(model_name)
    prompt = llama_prompt()
    parser = CustomSpaceSeparatedListOutputParser()
    prompt_and_model = prompt | llm | parser
    text = "Explain what are Deep Neural Networks in 2-3 sentences"
    print(prompt_and_model.invoke({"text" : text}))
