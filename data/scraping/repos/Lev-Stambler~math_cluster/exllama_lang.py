# From https://pastebin.com/p9KwXSSD
HF_AVAILABLE = False
try:
    from huggingface_hub import (
        try_to_load_from_cache, 
        snapshot_download, 
        _CACHED_NO_EXIST
    )
except ModuleNotFoundError:
    HF_AVAILABLE = False

from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from langchain.llms.utils import enforce_stop_tokens
from pydantic import Extra, Field, root_validator
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
from langchain.llms.base import LLM
from functools import partial
import torch
import glob
import sys
import os

from pprint import pprint as pp

# DEFAULT_PATH = "/home/john/Projects/Python/text-models/text-generation-webui/models/TheBloke_guanaco-65B-GPTQ/"
#EXLLAMA_PATH = '/home/john/Projects/Python/text-models/text-generation-webui/models/TheBloke_WizardLM-30B-Uncensored-GPTQ/'
#EXLLAMA_PATH = "/home/john/Projects/Python/text-models/text-generation-webui/models/TheBloke_alpaca-lora-65B-GPTQ-4bit/"

START = "\033[1;94m"
STOP = "\033[;97m"

# TODO: Make this whole thing better...
# PATH_TEMPLATE = "/home/john/Projects/Python/text-models/text-generation-webui/models/{model_name}/"

def clear_screen():
    print("\033c", end="")
    #print(chr(27) + "[2J")

class ExLLamaLLM(LLM):
    model_name: str = Field(None, alias='model_name')
    model_dir: str = Field(None, alias='model_dir')
    model_path: str = Field(None, alias='model_path')
    tokenizer_path: str = Field(None, alias='tokenizer_path')
    config_path: str = Field(None, alias='config_path')

    max_seq_len: int = Field(2048, alias='max_seq_len')

    temperature: float = Field(0.95, alias='temperature')
    top_k: int = Field(20, alias='top_k')
    top_p: float = Field(0.65, alias='top_p')
    min_p: float = Field(0.00, alias='min_p')
    
    token_repetition_penalty_max: float = Field(1.15, alias='token_repetition_penalty_max')
    token_repetition_penalty_sustain: int = Field(256, alias='token_repetition_penalty_sustain')
    token_repetition_penalty_decay: int = Field(None, alias='token_repetition_penalty_sustain')

    beams: int = Field(1, alias='beams')
    beam_length: int = Field(1, alias='beam_length')

    min_response_tokens: int = Field(4, alias='min_response_tokens')
    max_response_tokens: int = Field(384, alias='max_response_tokens')

    gpu_mem_allocation: str = Field(None, alias='gpu_mem_allocation')
    gpu_peer_fix: bool = Field(True, alias='gpu_peer_fix')

    stream_output: bool = Field(False, alias='stream_output')
    
    @property
    def _llm_type(self) -> str:
        """Return the type of llm."""
        return "exllama"

    @root_validator
    def validate_environment(cls, values: dict) -> dict:
        if values['token_repetition_penalty_decay'] is None:
            values['token_repetition_penalty_decay'] = values['token_repetition_penalty_sustain'] // 2

        values['llm'], values['tokenizer'] = cls._get_llm(values)

        return values

    @torch.no_grad()
    def _call(
        self,
        prompt: str,
        stop: None, # TODO: make modular
        run_manager: None,
    ) -> str:
        # print("AAAAAAAAAAAAAAAAAA", stop)
        if self.stream_output:
            #clear_screen()
            print(prompt + START, end=" ")

        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)
        
        ids = self.tokenizer.encode(prompt)
        remaining_tokens = self.max_seq_len - len(ids[0]) - 1
        if remaining_tokens <= 1:
            return ''
        self.llm.gen_begin(ids)
        self.llm.begin_beam_search()
        num_res_tokens = 0
        res_line = ""
        out_text = ""
        #print(f"========== REMAINING TOKENS: {remaining_tokens}")
        #pp(ids)
        max_response_tokens = self.max_response_tokens
        if max_response_tokens > remaining_tokens:
            max_response_tokens = remaining_tokens
        for i in range(max_response_tokens):

            # Disallowing the end condition tokens seems like a clean way to force longer replies.

            if i < self.min_response_tokens:
                self.llm.disallow_tokens([self.tokenizer.newline_token_id, self.tokenizer.eos_token_id])
            else:
                self.llm.disallow_tokens(None)

            # Get a token

            gen_token = self.llm.beam_search()

            # If token is EOS, replace it with newline before continuing

            if gen_token.item() == self.tokenizer.eos_token_id:
                self.llm.replace_last_token(self.tokenizer.newline_token_id)

            num_res_tokens += 1
            text = self.tokenizer.decode(self.llm.sequence_actual[:, -num_res_tokens:][0])
            new_text = text[len(res_line):]

            skip_space = res_line.endswith("\n") and new_text.startswith(" ")  # Bit prettier console output
            res_line += new_text

            if self.stream_output:
                print(new_text, end="")  # (character streaming output is here)
                sys.stdout.flush()

            if text_callback:
                text_callback(new_text)
            out_text += new_text

            if stop is not None:
                new_out_text = enforce_stop_tokens(out_text, stop)
                if new_out_text != out_text:
                    out_text = new_out_text
                    break

        self.llm.end_beam_search()

        if self.stream_output:
            print(STOP + '\n')
        return out_text.strip()

    def _get_llm(config_dict):
        config_path = config_dict['config_path']
        tokenizer_path = config_dict['tokenizer_path']
        model_path = config_dict['model_path']

        model_name = config_dict['model_name']
        model_dir = config_dict['model_dir']

        if model_dir is None:
            if HF_AVAILABLE:
                if model_name is not None:
                    snapshot_download(repo_id=model_name)
                    if config_path is None:
                        config_path = try_to_load_from_cache(model_name, 'config.json')
                        model_dir = os.path.dirname(config_path)
                    if tokenizer_path is None:
                        tokenizer_path = try_to_load_from_cache(model_name, 'tokenizer.model')
        else:
            if config_path is None:
                config_path = os.path.join(model_dir, 'config.json')
            if tokenizer_path is None:
                tokenizer_path = os.path.join(model_dir, 'tokenizer.model')

        if model_path is None and model_dir is not None:
            st_pattern = os.path.join(model_dir, "*.safetensors")
            st = glob.glob(st_pattern)
            if len(st) == 0:
                print(f" !! No files matching {st_pattern}")
                sys.exit()
            if len(st) > 1:
                print(f" !! Multiple files matching {st_pattern}")
                sys.exit()
            model_path = st[0]

        #model_name = config_dict['model_name'].replace('/', '_')
        #model_dir = PATH_TEMPLATE.format(model_name=model_name)

        config = ExLlamaConfig(config_path)
        config.model_path = model_path

        # TODO: Make these configurable
        #config.attention_method = ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP
        #config.matmul_method = ExLlamaConfig.MatmulMethod.SWITCHED
        #config.mlp_method = ExLlamaConfig.MLPMethod.NORMAL
        config.stream_layer_interval = 0

        config.gpu_peer_fix = config_dict['gpu_peer_fix']
        config.max_seq_len = config_dict['max_seq_len']

        if config_dict['gpu_mem_allocation']:
            config.set_auto_map(config_dict['gpu_mem_allocation'])

        #config.set_dequant(None)

        model = ExLlama(config)
        cache = ExLlamaCache(model)
        tokenizer = ExLlamaTokenizer(tokenizer_path)

        generator = ExLlamaGenerator(model, tokenizer, cache)
        generator.settings = ExLlamaGenerator.Settings()
        
        generator.settings.temperature = config_dict['temperature']
        generator.settings.top_k = config_dict['top_k']
        generator.settings.top_p = config_dict['top_p']
        generator.settings.min_p = config_dict['min_p']
        generator.settings.token_repetition_penalty_max = config_dict['token_repetition_penalty_max']
        generator.settings.token_repetition_penalty_sustain = config_dict['token_repetition_penalty_sustain']
        generator.settings.token_repetition_penalty_decay = config_dict['token_repetition_penalty_decay']
        generator.settings.beams = config_dict['beams']
        generator.settings.beam_length = config_dict['beam_length']

        return generator, tokenizer

