from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig)

import weaviate
from sentence_transformers import SentenceTransformer
from prompts import OPENAI_PROMPT_TEMPLATE,SIMPLE_PREDICTION_OPENAI_PROMPT_TEMPLATE, SIMPLE_PREDICTION_OPENAI_PROMPT_TEMPLATE, RETRIEVER_PROMPT_TEMPLATE
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai

# Load the Lora model


# we could for starters tell the user to b as detailed with their request as they can
class Models:
    def __init__(self):
        use_4bit = True
        bnb_4bit_compute_dtype = "float16"
        bnb_4bit_quant_type = "nf4"
        use_nested_quant = False

        bnb_config = BitsAndBytesConfig(
            load_in_4bit = use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant = use_nested_quant,
        )
        peft_model_id = "odunola/bloomz_reriever_instruct"
        config = PeftConfig.from_pretrained(peft_model_id)
        rephrase_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, quantization_config = bnb_config, device_map = 'auto')
        self.tokenizer2 = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        self.llm2 = PeftModel.from_pretrained(rephrase_model, peft_model_id)
        self.semantic_model = SentenceTransformer('thenlper/gte-large')
        self.client = weaviate.Client(
                url="https://testing-area-4ps7dhgv.weaviate.network", #for testing
            )
        

    async def retrieve(self, query: str) -> list:#this gets the context from the vector db
        prompt = f"To generate a representation for this sentence for use in retrieving related articles: {query}"
        query_vector = self.semantic_model.encode(prompt)
        response = self.client.query.get(
        "Recipes",
        ["texts"]
            ).with_limit(3).with_near_vector(
                {'vector': query_vector}
            ).do()
        res = response['data']['Get']['Recipes']
        return [i['texts'] for i in res]



    async def reply(self, query:str, contexts: list) -> str:
        context_str = "\n".join(contexts)
        user_prompt = OPENAI_PROMPT_TEMPLATE.format(context_str = context_str,query =query)
        chat_completion = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=user_prompt, max_tokens = 500, temperature = 0.7)
        response = chat_completion['choices'][0]['text']
        return response
    

    async def predict(self, text: str) -> str:
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"system", "content":SIMPLE_PREDICTION_OPENAI_PROMPT_TEMPLATE},{"role": "user", "content": text}])
        response = chat_completion['choices'][0]['message']['content']
        return response
    
    async def generate_interpretation(self,text:str) -> str:
            prompt = RETRIEVER_PROMPT_TEMPLATE.format(request = text)
            tokens = self.tokenizer2(prompt, return_tensors = 'pt')
            outputs =  self.llm2.generate(input_ids = tokens['input_ids'].to('cuda'), temperature = 0.2, max_length = 200, do_sample = True)
            response = self.tokenizer2.decode(outputs[0], skip_special_tokens=True).split('Interpretation:')[-1]       
            return response
            


