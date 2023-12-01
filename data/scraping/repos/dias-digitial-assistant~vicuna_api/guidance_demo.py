from transformers import AutoModelForCausalLM, AutoTokenizer
import guidance.llms

class AutoLLM(guidance.llms.Transformers):
    cache = guidance.llms.LLM._open_cache("_auto.diskcache")

    def __init__(self, model, tokenizer=None, device_map=None, **kwargs):
        """ Create a new auto model.
        """
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model,
            device_map="auto",
            load_in_8bit=True,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            load_in_8bit=True,
        )

        super().__init__(model, tokenizer=tokenizer, device_map=device_map, **kwargs)
        
llm = AutoLLM(model=PATH_TO_VICUNA,device_map="auto")
prompt = guidance('''The link is <a href="http:{{gen max_tokens=10 token_healing=False}}''')
prompt(llm = llm)
