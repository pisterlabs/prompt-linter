import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from guidance.llms import Transformers
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from model_roles import get_role_from_model_name

if torch.cuda.is_available() and not torch.version.hip:
    try:
        # from llama_cpp_cuda import Llama
        # from llama_cpp_python import Llama
        from llama_cpp import Llama
    except:
        print("Failed to import llama_cpp_cuda, falling back on llama_cpp")
        from llama_cpp import Llama
else:
    from llama_cpp import Llama

selected_role = None


def get_role():
    return selected_role


class LlamacppHF(Transformers):
    llm_name: str = "llama"

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):
        assert tokenizer is None, "We will not respect any tokenizer from the caller."
        assert isinstance(model, str), "Model should be a str"

        global selected_role
        selected_role = get_role_from_model_name(model)

        print(f"Initializing LlamacppHF with model {model}")

        tokenizer = AutoTokenizer.from_pretrained("TheBloke/StableBeluga-13B-GPTQ")

        model = LlamacppHFInner.from_pretrained(model)

        # model._update_model_kwargs_for_generation = (
        #     LlamaForCausalLM._update_model_kwargs_for_generation
        # )
        # model.config.max_seq_len = 4096  # this is the one

        return super()._model_and_tokenizer(model, tokenizer, **kwargs)

    @staticmethod
    def role_start(role):
        return get_role().role_start(role)

    @staticmethod
    def role_end(role):
        return get_role().role_end(role)


class LlamacppHFInner(PreTrainedModel):
    def __init__(self, model):
        super().__init__(PretrainedConfig())
        self.model = model
        self.generation_config = GenerationConfig()
        self.cache = None

        # sometimes this manually has to change.
        # some models expect i.e. 32002, others 32000.
        self.config.vocab_size = 32000

        self.past_seq = None
        self.llamacpp_cache = {
            'n_tokens': self.model.n_tokens,
            'input_ids': self.model.input_ids,
            'scores': self.model.scores,
            'ctx': self.model.ctx
        }

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def load_cache(self):
        self.model.n_tokens = self.llamacpp_cache['n_tokens']
        self.model.input_ids = self.llamacpp_cache['input_ids']
        self.model.scores = self.llamacpp_cache['scores']
        self.model.ctx = self.llamacpp_cache['ctx']

    def save_cache(self):
        self.llamacpp_cache.update({
            'n_tokens': self.model.n_tokens,
            'input_ids': self.model.input_ids,
            'scores': self.model.scores,
            'ctx': self.model.ctx
        })

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        use_cache = kwargs.get('use_cache', True)
        labels = kwargs.get('labels', None)
        past_key_values = kwargs.get('past_key_values', None)

        if len(args) > 0:
            print('assuming cfg-cache option for llamacpp_HF')

            input_ids = args[0]
            is_negative = True
            past_seq = self.past_seq_negative
            self.load_negative_cache()
        else:
            input_ids = kwargs['input_ids']
            is_negative = False
            past_seq = self.past_seq
            self.load_cache()

        seq = input_ids[0].tolist()
        if is_negative and past_key_values is not None:
            seq = past_key_values + seq

        seq_tensor = torch.tensor(seq)

        # Make the forward call
        if labels is None:
            if past_seq is None or not torch.equal(past_seq, seq_tensor[:-1]):
                self.model.reset()
                self.model.eval(seq)
            else:
                self.model.eval([seq[-1]])

            logits = torch.tensor(self.model.scores[self.model.n_tokens - 1, :]).view(1, 1, -1).to(input_ids.device)
        else:
            self.model.reset()
            self.model.eval(seq)
            logits = torch.tensor(self.model.eval_logits)
            logits = logits.view(1, logits.shape[0], logits.shape[1]).to(input_ids.device)

        if is_negative:
            self.save_negative_cache()
            self.past_seq_negative = seq_tensor
        else:
            self.save_cache()
            self.past_seq = seq_tensor

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(logits=logits, past_key_values=seq if use_cache else None, loss=loss)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        assert (
            len(model_args) == 0 and len(kwargs) == 0
        ), "extra args is currently not supported"
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        path = Path(pretrained_model_name_or_path)
        if path.is_file():
            model_file = path
        else:
            model_file = list(path.glob("*.gguf"))[0]

        params = {
            "model_path": str(model_file),
            "n_ctx": 4096,
            "seed": 0,
            "n_threads": 15,
            "n_batch": 512,
            "use_mmap": True,
            "use_mlock": False,
            "low_vram": False,
            "mul_mat_q": False,
            "n_gpu_layers": 41,
            # "rope_freq_base": 10000 * shared.args.alpha_value ** (64 / 63.0),
            # "rope_freq_scale": 1.0 / shared.args.compress_pos_emb,
            # "n_gqa": shared.args.n_gqa or None,
            "rms_norm_eps": 0.000005,
            "logits_all": True,
        }

        # Llama = llama_cpp_lib().Llama
        model = Llama(**params)

        return LlamacppHFInner(model)
