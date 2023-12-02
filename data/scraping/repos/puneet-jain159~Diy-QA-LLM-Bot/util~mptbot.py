"""Wrapper around HuggingFace Pipeline APIs."""
import importlib.util
import logging
from typing import Any, List, Mapping, Optional

from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

DEFAULT_MODEL_ID = "mosaicml/mpt-30b-chat"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text-generation")



class HuggingFacePipelineLocal(LLM):

    pipeline: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    model_kwargs: Optional[dict] = None
    tokenizer: Any
    config: Any
    """Key word arguments passed to the model."""
    pipeline_kwargs: Optional[dict] = None
    """Key word arguments passed to the pipeline."""
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str,
        device_map: str = 'auto',
        trust_remote_code: bool = True,
        model_kwargs: Optional[dict] = None,
        pipeline_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""
        try:
            import torch
            import transformers
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                AutoConfig

            )
            from huggingface_hub import snapshot_download
            from transformers import pipeline as hf_pipeline

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )
        try :
          print("Checking if the model has been already downloaded ....")
          model_dir =snapshot_download(model_id,
                    # revision = revision,
                    resume_download=True,
                    local_files_only = True)
        except:
          print("Load the new model ....")
          model_dir =snapshot_download(model_id,
          # revision = revision,
          resume_download=True,
          local_files_only = False)
        _model_kwargs = model_kwargs or {}

        if "load_in_8bit" not in _model_kwargs:
          _model_kwargs["load_in_8bit"] =  True


        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                  trust_remote_code=trust_remote_code,
                                                  padding = "left" )


        _pipeline_kwargs = pipeline_kwargs or {}


        config_lm = AutoConfig.from_pretrained(model_id,
                                    trust_remote_code=True)
        
        if "mpt-30b" in model_id:
          config_lm.max_seq_len = 16384
          config_lm.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
          config_lm.init_device = 'cuda:0'
          torch_dtype = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                config=config_lm,
                torch_dtype=torch_dtype, # Load model weights in bfloat16
                trust_remote_code=trust_remote_code,
                # revision=revision,
                device_map = 'auto',
                **model_kwargs)

        return cls(
            pipeline=model,
            model_id=model_id,
            config =config_lm,
            tokenizer = tokenizer,
            model_kwargs=_model_kwargs,
            pipeline_kwargs=_pipeline_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
            "pipeline_kwargs": self.pipeline_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_mpt_pipeline"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response = self.mpt_instruct_generate(prompt)
        return response
    

    def mpt_instruct_generate(self,prompt):
        '''
        Def mpt_instruct_generate
        '''
        from transformers import pipeline as hf_pipeline
        import torch

        if 'max_new_tokens' not in self.pipeline_kwargs:
          self.pipeline_kwargs['max_new_tokens'] = 256

        if 'temperature' not in self.pipeline_kwargs:
          self.pipeline_kwargs['temperature'] = 0.15

        if 'top_k' not in self.pipeline_kwargs:
          self.pipeline_kwargs['top_k'] = 0

        if 'top_p' not in self.pipeline_kwargs:
          self.pipeline_kwargs['top_p'] = 1
        if 'eos_token_id' not in self.pipeline_kwargs:
          self.pipeline_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
          self.pipeline_kwargs['pad_token_id'] = self.tokenizer.eos_token_id
        
        if 'do_sample' not in self.pipeline_kwargs:
          self.pipeline_kwargs['do_sample'] = True
        
        self.pipeline_kwargs['use_cache'] = True

        # if 'num_return_sequences' not in self.pipeline_kwargs:
        #   self.pipeline_kwargs['num_return_sequences'] = 1
        # print("self.pipeline_kwargs:" ,self.pipeline_kwargs)
        # print("Prompt" ,[prompt])

        generator = hf_pipeline("text-generation",
                            model=self.pipeline, 
                            config=self.config, 
                            tokenizer=self.tokenizer,
                            torch_dtype=torch.bfloat16)
          
          
        if "return_full_text" not in self.pipeline_kwargs:
            self.pipeline_kwargs["return_full_text"] = False

      
        if isinstance(prompt, str):
          # print("prompt:...", prompt)
          generated_text = generator(prompt, **self.pipeline_kwargs)[0]["generated_text"]
          # print("generated_text:...", generated_text)
        elif isinstance(prompt, list):
          outputs = generator(prompt, **self.pipeline_kwargs)
          generated_text = [out[0]["generated_text"] for out in outputs]
        return generated_text