
from typing import Any, List, Mapping, Optional

from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

DEFAULT_MODEL_ID = "gpt2"


class OpenVINO_Pipeline(LLM):
    """Wrapper around the OpenVINO model"""

    model_id: str = DEFAULT_MODEL_ID
    """Model name or model path to use."""
    model_kwargs: Optional[dict] = None
    model: Any  #: :meta private:
    """LLM Transformers model."""
    tokenizer: Any  #: :meta private:
    """Huggingface tokenizer model."""
    streaming: bool = True
    """Whether to stream the results, token by token."""
    max_new_tokens: int = 64
    """Maximum number of new token generated."""
    
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        model_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """
        Construct object from model_id
        
        Args:
        
            model_id: Path for the huggingface repo id to be downloaded or
                      the huggingface checkpoint folder.
            model_kwargs: Keyword arguments that will be passed to the model and tokenizer.
            kwargs: Extra arguments that will be passed to the model and tokenizer.

        Returns: An object of TransformersLLM.
        """
        try:
            from optimum.intel.openvino import OVModelForCausalLM
            from transformers import AutoTokenizer, LlamaTokenizer

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )

        _model_kwargs = model_kwargs or {}
        # TODO: may refactore this code in the future
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **_model_kwargs)
        except:
            tokenizer = LlamaTokenizer.from_pretrained(model_id, **_model_kwargs)

        # TODO: may refactore this code in the future
        try:
            model = OVModelForCausalLM.from_pretrained(model_id, compile=False, **_model_kwargs)
        except:
            model = OVModelForCausalLM.from_pretrained(model_id, compile=False, export=True, **_model_kwargs)
            
        model.compile()

        if "trust_remote_code" in _model_kwargs:
            _model_kwargs = {
                k: v for k, v in _model_kwargs.items() if k != "trust_remote_code"
            }

        return cls(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            model_kwargs=_model_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "OpenVINO backend"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:
            from transformers import TextStreamer
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            if stop is not None:
                from transformers.generation.stopping_criteria import StoppingCriteriaList
                from transformers.tools.agents import StopSequenceCriteria
                stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(stop,
                                                                               self.tokenizer)])
            else:
                stopping_criteria = None
            output = self.model.generate(input_ids, streamer=streamer, max_new_tokens=self.max_new_tokens,
                                         stopping_criteria=stopping_criteria, **kwargs)
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return text
        else:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            if stop is not None:
                from transformers.generation.stopping_criteria import StoppingCriteriaList
                from transformers.tools.agents import StopSequenceCriteria
                stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(stop,
                                                                               self.tokenizer)])
            else:
                stopping_criteria = None
            output = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens, stopping_criteria=stopping_criteria, **kwargs)
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt) :]
            return text