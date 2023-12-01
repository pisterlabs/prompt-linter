import transformers
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM


class Llama2QLLM():
    def __init__(self, verbose=False):
        self.model_id = "TheBloke/vicuna-13B-v1.5-GPTQ"
        self.model_basename = "gptq_model-4bit-128g"
        self.verbose = verbose
        self.pipeline, self.bare_model, self.tokenizer = self.load_pipeline()
        self.chat_template = """
            Floodbrain is a helpful assistant that is able to answer questions about flood related natural disasters.

            Question: {query_str}
            Response: """

        self.blank_template = """
            {query_str}
            """

    def __call__(self, prompt):
        prompt = self.blank_template.format(query_str=prompt)
        response = self.pipeline(prompt)
        return response

    def load_pipeline(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        bare_model = AutoGPTQForCausalLM.from_quantized(
            self.model_id,
            model_basename=self.model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            device_map="balanced_low_0",
            use_triton=False,
            quantize_config=None,

        )

        pipeline = transformers.pipeline(
            "text-generation",
            model=bare_model,
            tokenizer=tokenizer,
            temperature=0.1,
            max_new_tokens=1024,
            top_p=0.95,
            repetition_penalty=1.15
        )

        pipeline = HuggingFacePipeline(pipeline=pipeline, verbose=self.verbose)
        return pipeline, bare_model, tokenizer


if __name__ == '__main__':
    llm_model = Llama2QLLM(verbose=True)

    # example function call would be
    prompt = "Any weather events in the next 24 hours you think we should be aware of? For example, do any cities look like they might be in risk of flooding?"
    response = llm_model(prompt)
    print('Question:', prompt)
    print("Response:", response)