import torch
from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding
from llama_index import SimpleDirectoryReader, LLMPredictor, PromptHelper, GPTSimpleVectorIndex
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


hf_model_path = "models/llama-7b"
alpaca_model_path = "models/lora-alpaca"

tokenizer = LlamaTokenizer.from_pretrained(hf_model_path)

model = LlamaForCausalLM.from_pretrained(
    hf_model_path,
    load_in_8bit=True, #Dissabling could solve some errors
    device_map="auto",
)
model = PeftModel.from_pretrained(model, alpaca_model_path)


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
max_length = 1500 #2048
max_new_tokens = 48


class LLaMALLM(LLM):
    def _call(self, prompt, stop=None):
        prompt += "### Response:"

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        
        generation_config = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.15,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=128,
            )
        response = ""
        for s in generation_output.sequences:
            response += tokenizer.decode(s)
            
        response = response[len(prompt):]
        print("Model Response:", response)
        return response

    def _identifying_params(self):
        return {"name_of_model": "alpaca"}

    def _llm_type(self):
        return "custom"

max_input_size = max_length
num_output = max_new_tokens
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
documents = SimpleDirectoryReader('data').load_data()
llm_predictor = LLMPredictor(llm=LLaMALLM())
index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper)

index.save_to_disk('index.json')
new_index = GPTSimpleVectorIndex.load_from_disk('index.json', embed_model=embed_model, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

response = new_index.query("What did Gatsby do before he met Daisy?")
print(response.response)

response = new_index.query("What did the narrator do after getting back to Chicago?")
print(response.response)
