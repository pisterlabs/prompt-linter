"""
This module is adapted from
https://huggingface.co/mosaicml/mpt-7b/discussions/16
"""

from functools import partial
from threading import Thread
from typing import Any, Dict, List, Mapping, Optional, Set

import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from pydantic import Extra, Field, root_validator
from transformers import AutoTokenizer


class WizardLM(LLM):
    model_name: str = Field("TheBloke/WizardLM-30B-GPTQ",
                            alias='model_name')
    """The name of the model to use."""

    model_basename: str = Field("wizardlm-30b-GPTQ-4bit.act.order",
                                alias="model_basename")

    tokenizer_name: str = Field("TheBloke/WizardLM-30B-GPTQ",
                                alias='tokenizer_name')
    """The name of the sentence tokenizer to use."""

    config: Any = None  #: :meta private:
    """The reference to the loaded configuration."""

    tokenizer: Any = None  #: :meta private:
    """The reference to the loaded tokenizer."""

    model: Any = None  #: :meta private:
    """The reference to the loaded model."""

    stop: Optional[List[str]] = []
    """A list of strings to stop generation when encountered."""

    temperature: Optional[float] = Field(0.8, alias='temperature')
    """The temperature to use for sampling."""

    max_new_tokens: Optional[int] = Field(512, alias='max_new_tokens')
    """The maximum number of tokens to generate."""

    top_p: Optional[float] = Field(0.95, alias='top_p')
    repetition_penalty: Optional[float] = Field(1.15,
                                                alias='repetition_penalty')
    skip_special_tokens: Optional[bool] = Field(True,
                                                alias='skip_special_tokens')

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _wizard_default_params(self) -> Dict[str, Any]:
        """Get the default parameters."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "skip_special_tokens": self.skip_special_tokens
        }

    @staticmethod
    def _wizard_param_names() -> Set[str]:
        """Get the identifying parameters."""
        return {
            "max_new_tokens",
            "temperature",
            "top_p",
            "repetition_penalty",
            "skip_special_tokens"
        }

    @staticmethod
    def _model_param_names(model_name: str) -> Set[str]:
        """Get the identifying parameters."""
        # TODO: fork for different parameters for different model variants.
        return WizardLM._wizard_param_names()

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters."""
        return self._wizard_default_params()

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate the environment."""
        try:
            model = AutoGPTQForCausalLM.from_quantized(values["model_name"],
                                                       model_basename=values["model_basename"],
                                                       use_safetensors=True,
                                                       trust_remote_code=True,
                                                       device="cuda:0",
                                                       use_triton=False,
                                                       quantize_config=None)
            tokenizer = AutoTokenizer.from_pretrained(values["model_name"],
                                                      use_fast=True)

            values["model"] = model
            values["tokenizer"] = tokenizer

        except Exception as e:
            raise Exception(f"WizardLM failed to load with error: {e}")
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model_name,
            **self._default_params,
            **{
                k: v
                for k, v in self.__dict__.items()
                if k in self._model_param_names(self.model_name)
            },
        }

    @property
    def _llm_type(self) -> str:
        """Return the type of llm."""
        return "wizardlm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        r"""Call out to WizardLM's generate method via transformers.

        Args:
            prompt: The prompt to pass into the model.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                prompt = "This is a story about a big sabre tooth tiger: "
                response = model(prompt)
        """
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token,
                                    verbose=self.verbose)
        text = ""
        model = self.model
        tokenizer = self.tokenizer

        prompt_template = '''A chat between a curious user and an artificial intelligence assistant.
        The assistant gives helpful, detailed, and polite answers to the user's questions.
        USER: {prompt}
        ASSISTANT:'''

        inputs = tokenizer([prompt_template.format(prompt=prompt)],
                           return_tensors="pt").input_ids.cuda()

        gen_ids = model.generate(inputs=inputs,
                                 temperature=self.temperature,
                                 max_new_tokens=self.max_new_tokens,
                                 top_p=self.top_p,
                                 repetition_penalty=self.repetition_penalty)

        output = tokenizer.batch_decode(gen_ids,
                                        skip_special_tokens=self.skip_special_tokens)        

        return [x[x.find('ASSISTANT:')+10:] for x in output][0]


if __name__ == '__main__':
    # llm = MLegoLLM()
    llm = WizardLM()

    text = """
    Summarize below text into 15 words:


    Truly unprecedented. An AI discovery that, weirdly enough, will make Elon Musk as much excited as you… while leaving Hollywood extremely worried.

    In my opinion, this is potentially more significant than GPT-4, considering the novelty of the field and the potential for disruption.

    Cutting to the chase, NVIDIA and the creators of Stable Diffusion have presented VideoLDM, a new state-of-the-art video synthesis model that proves, once again, that the world will never be the same after AI.

    It can go for minutes long with entirely made-up scenery and interactions.
    It can recreate multiple scenarios at wish, with no human interference.
    Entirely by itself.

    The first text-to-video high-quality generator.

    Let’s understand how humans managed to create this… “thing”, how does it work by showing examples, and, above all… should we be afraid?

    """
    print(llm(text))
