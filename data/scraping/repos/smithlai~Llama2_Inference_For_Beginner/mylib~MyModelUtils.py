from torch import cuda, bfloat16
import transformers
from langchain.llms import HuggingFacePipeline

class MyModelUtils:
    @classmethod
    def device(cls):
        return f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    def __init__(self, model_id):
        self.model_id = model_id
        self.default_modelconf_kwargs={k: v for k, v in {
            'do_sample':True,
            # stopping_criteria:stopping_criteria,  # without this model rambles during chat
            'temperature':0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            'max_new_tokens':256,  # max number of tokens to generate in the output
            'repetition_penalty':1.1,  # without this output begins repeating
            # top_k 
            # top_p 
        }.items() if v is not None}
                
    def make_bnb_config(self):
        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        return bnb_config
    def make_modelcfg_kwargs(self, **modelconf_kwargs):
        # https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/configuration#transformers.PretrainedConfig

        modelconf_kwargs = self.default_modelconf_kwargs | modelconf_kwargs  # NOTE: 3.9+ ONLY
        return transformers.AutoConfig.from_pretrained(self.model_id, **modelconf_kwargs)
    # def make_stop(self, tokenizer):
    #     import torch
    #     stop_list = ['\nHuman:', '\n```\n']
        
    #     stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    #     print(stop_token_ids)
        
    #     stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
    #     print(stop_token_ids)
    
    #     from transformers import StoppingCriteria, StoppingCriteriaList
    
    #     # define custom stopping criteria object
    #     class StopOnTokens(StoppingCriteria):
    #         def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    #             for stop_ids in stop_token_ids:
    #                 if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
    #                     return True
    #             return False
    
    #     stopping_criteria = StoppingCriteriaList([StopOnTokens()])
    #     return stopping_criteria

    # AutoModel.from_pretrained : https://huggingface.co/docs/transformers/v4.33.0/en/model_doc/auto#transformers.AutoModel.from_pretrained
    # AutoModelForCausalLM.from_pretrained : https://huggingface.co/docs/transformers/v4.33.0/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained
    # args for from_pretrained(.... ,**kwargs)
    def make_model_kwargs_for_pretrained(self, **model_conf_kwargs):
        default_model_kwargs ={k: v for k, v in { 
            'device_map':'auto', 
            'config': self.make_modelcfg_kwargs(**model_conf_kwargs),
            'quantization_config':self.make_bnb_config(),
        }.items() if v is not None}
        return default_model_kwargs
    
    def init_model(self, **model_kwargs) -> transformers.PreTrainedModel:
        model_conf_kwargs={}
        # print(model_kwargs)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        model.eval()
        return model

    def init_hf_pipeline(self, pretrained_kwargs, **pipeline_kwargs):
        pretrained_kwargs = pretrained_kwargs | {}
        # https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
        # Do not use device_map AND device at the same time as they will conflict
        hf_model = self.init_model(**pretrained_kwargs)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, **pretrained_kwargs)
        hf_pipeline = transformers.pipeline(
            model=hf_model, 
            tokenizer = tokenizer,
            return_full_text=True,  # 就是是否連input也重覆輸出，langchain expects the full text
            task='text-generation',
            # # =========================
            # do_sample=True,
            # temperature=0.01,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            # max_new_tokens=256,  # max number of tokens to generate in the output
            # repetition_penalty=1.1  # without this output begins repeating
        )
        return hf_pipeline
    
