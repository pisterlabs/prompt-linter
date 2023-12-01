#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Tannon Kew'
# __Email__ = 'kew@cl.uzh.ch
# __Date__ = '2023-03-03'

import os
import sys
import math
import time
import json
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import torch
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    LlamaForCausalLM,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
)
from accelerate import init_empty_weights, infer_auto_device_map

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512mb"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

@dataclass
class InferenceArguments:
    """
    Arguments pertaining to running generation/inference with pre-trained/fine-tuned model.
    """

    ################ 
    ## model loading
    ################

    model_name_or_path: str = field(
        default="",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    is_encoder_decoder: bool = field(
        default=False,
        metadata={"help": "If set to True, will load a Encoder-Decoder Model instead of Decoder-only."}
    )

    # checkpoint_dir: str = field(
    #     default=None,
    #     metadata={"help": "Path to fine-tuned model checkpoint"}
    # )
    
    load_in_8bit: bool = field(
        default=True,
        metadata={"help": "If set to True, model will be loaded with int8 quantization "
                          "(see https://huggingface.co/blog/hf-bitsandbytes-integration)"}
    )

    offload_state_dict: bool = field(
        default=True,
        metadata={"help": "Whether to offload state dict (useful for very large LMs)"}
    )

    offload_folder: str = field(
        default="resources/offload/",
        metadata={"help": "directory path for offloading"}
    )

    device_map: str = field(
        default="auto",
        metadata={"help": ""}
    )

    max_memory: float = field(
        default=0.95,
        metadata={"help": "Maximum allowable GPU memory useage for each GPU"}
    )

    ###################
    ## inference params
    ###################

    seed: int = field(
        default=42,
        metadata={"help": "random seed"}
    )

    use_cuda: bool = field(
        default=True,
        metadata={"help": "Use GPU if available"}
    )

    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size for predictions"}
    )

    min_length: int = field(
        default=None,
        metadata={"help": "Minimum length of generated text"}
    )

    max_new_tokens: int = field(
        default=100,
        metadata={"help": "Maximum number of tokens to generate"}
    )

    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Length penalty for generated text"}
    )

    no_early_stop: bool = field(
        default=False,
        metadata={"help": "Disable early stopping on generate"}
    )

    num_return_sequences: int = field(
        default=1,
        metadata={"help": "Number of sequences to generate"}
    )

    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search"}
    )

    do_sample: bool = field(
        default=True,
        metadata={"help": "Sample instead of greedy decoding"}
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for generation"}
    )
    
    top_k: int = field(
        default=0,
        metadata={"help": "Number of top k tokens to keep for top-k sampling"}
    )

    top_p: float = field(
        default=0.9,
        metadata={"help": "Probability of top-p sampling"}
    )

    verbose: bool = field(
        default=False,
        metadata={"help": "Print progress"}
    )

    ###################
    ## API parameters
    ###################
    frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "Can be used to reduce repetitiveness of generated tokens. "
                          "Values should be between 0.0 and 1.0. "
                          "Similar to presence_penalty, but works on relative frequency."}
    )

    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Can be used to reduce repetitiveness of generated tokens. "
                          "Values should be between 0.0 and 1.0. "
                          "Similar to presence_penalty, but works on more immediate blocking."}
    )

    trial_key: bool = field(
        default=False,
        metadata={"help": "Set to True if using Cohere models with a trial key "
                          "to avoid hitting the rate limit of 5 requests per minute."}
    )

    ###################
    ## data and prompts
    ###################

    input_file: str = field(
        default=None,
        metadata={"help": "Input file containing source sentences"}
    )

    output_dir: str = field(
        default="resources/outputs/",
        metadata={"help": "Path to output directory"}
    )

    output_file: str = field(
        default=None,
        metadata={"help": "Output file for model generations"}
    )

    prompt_json: str = field(
        default=None,
        metadata={"help": "JSON file containing pre-defined prompt fields and values."}
    )

    prompt_prefix: str = field(
        default="This is a prompt:",
        metadata={"help": "Prefix for generation prompt. This is passed to LangChain."}
    )

    prompt_template: str = field(
        default=r"Complex: {complex}\nSimple: {simple}",
        metadata={"help": "Example for generation prompt. This is passed to LangChain."}
    )
    
    prompt_suffix: str = field(
        default=r"Complex: {input}\nSimple:",
        metadata={"help": "Suffix for generation prompt. This is passed to LangChain."}
    )

    prompt_format: str = field(
        default="prefix_initial",
        metadata={"help": "Format for generation prompt. Either `prefix_initial` or `prefix_every`. "
                          "See description in prompt_utils.py."}
    )

    example_selector: str = field(
        default="random",
        metadata={"help": "Method for selecting examples from examples file."}
    )

    example_selector_mode: str = field(
        default="min",
        metadata={"help": "Method for ordering examples used in a prompt (only used if example_selector is 'sem_sim')."}
    )

    example_selector_model_name: str = field(
        default="all-mpnet-base-v2",
        metadata={"help": "Name of SentenceTransformers model to use for semantic similarity (only used if example_selector is 'sem_sim')."}
    )

    example_selector_save_dir: str = field(
        default="", #"resources/embeddings",
        metadata={"help": "Path to stored embeddings (only used if example_selector is 'sem_sim')."}
    )

    example_separator: str = field(
        default=r"\n\n",
        metadata={"help": "Delimiter for prompts and generated text"}
    )
    
    source_field: str = field(
        default="complex",
        metadata={"help": "Field in input file containing source sentences"}
    )

    target_field: str = field(
        default="simple",
        metadata={"help": "Field in input file containing target sentences"}
    )

    few_shot_n: int = field(
        default=0,
        metadata={"help": "number of examples to use as few-shot in-context examples"}
    )

    n_refs: int = field(
        default=1,
        metadata={"help": "Number of target reference examples to show for each few-shot demonstration."}
    )

    ref_delimiter: str = field(
        default=r"\t",
        metadata={"help": "Delimiter for multiple example references in prompt"}
    )

    examples: str = field(
        default=None,
        metadata={"help": "file containing examples for few-shot prompting, e.g. a validation/training dataset"}
    )


class LLM(object):
    """
    Custom class for loading and running language models from HuggingFace.
    """
    def __init__(self, args: InferenceArguments):
        # https://github.com/huggingface/accelerate/issues/864#issuecomment-1327726388    
        start_time = time.time()
        
        # set seed for reproducibility
        self.args = args
        
        if args.is_encoder_decoder:
            model_type = AutoModelForSeq2SeqLM
        else:
            model_type = AutoModelForCausalLM

        config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        
        with init_empty_weights():
            model = model_type.from_config(config)
            
            if torch.cuda.device_count() > 1: # set device map for multi-gpu ensuring no split modules
                if 'T5Block' in repr(model):
                    device_map = infer_auto_device_map(model, no_split_module_classes=["T5Block"], dtype=torch.float16, max_memory=self.set_max_memory())
                    device_map['lm_head'] = device_map["decoder.embed_tokens"] # https://github.com/akkikiki/huggingface_examples/blob/main/examples/load_flan_ul2.py#L15
                elif 'BloomBlock' in repr(model):
                    device_map = infer_auto_device_map(model, no_split_module_classes=["BloomBlock"], dtype=torch.float16, max_memory=self.set_max_memory())
                    device_map['lm_head'] = device_map["transformer.word_embeddings"]
                elif 'OPTDecoderLayer' in repr(model):
                    device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"], dtype=torch.float16, max_memory=self.set_max_memory())
                    device_map['lm_head'] = device_map["model.decoder.embed_tokens"] # https://github.com/akkikiki/huggingface_examples/blob/main/examples/load_opt.py#L18
                elif 'LlamaDecoderLayer' in repr(model):
                    device_map = infer_auto_device_map(model, no_split_module_classes=["LlamaDecoderLayer"], dtype=torch.float16, max_memory=self.set_max_memory())
                    # device_map['lm_head'] = device_map["model.embed_tokens"] # not required for Llama
                elif 'GPTNeoXLayer' in repr(model):
                    device_map = infer_auto_device_map(model, no_split_module_classes=["GPTNeoXLayer"], dtype=torch.float16, max_memory=self.set_max_memory())
                    # device_map['embed_out'] = device_map["gpt_neox.embed_in"] # not required for GPTNeoX
                else:
                    device_map = infer_auto_device_map(model, dtype=torch.float16, max_memory=self.set_max_memory())   
            else:
                device_map = "auto"

        logger.info(f"Device map: {device_map}")

        self.model = model_type.from_pretrained(
            self.args.model_name_or_path,
            device_map=device_map,
            load_in_8bit=self.args.load_in_8bit,
            torch_dtype=torch.float16,
            offload_state_dict=self.args.offload_state_dict,
            offload_folder=self.args.offload_folder,
            )
        end_time = time.time()
        logger.info(f"Loaded model {self.args.model_name_or_path} in {end_time - start_time:.4f} seconds")
        logger.info(f"Model parameters: {self.model.num_parameters():,} / footprint: {self.model.get_memory_footprint() / (1024*1024*1024):.2f} GB")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, padding_side='left')

        # LLAMA models don't have a pad token, so add one and resize the embeddings to match
        if self.tokenizer.pad_token_id is None: 
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            # self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.decode([self.model.config.pad_token_id])})

    def set_max_memory(self) -> Optional[Dict[int, str]]:
        n_gpus = torch.cuda.device_count()
        if self.args.max_memory and n_gpus > 1:
            logger.info(f"Infering max memory...")
            t = torch.cuda.get_device_properties(0).total_memory / (1024*1024*1024)
            # note, we use math.floor() as a conservative rounding method
            # to optimize the maximum batch size on multiple GPUs, we give the first GPU less memory
            # see max_memory at https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling
            max_memory = {
                i: (f"{math.floor(t*self.args.max_memory)}GiB" if i > 0 else
                    f"{math.floor(t*self.args.max_memory*0.3)}GiB") for i in range(n_gpus)
                }
            # max_memory['cpu'] = '400GiB' # may need to lower this depending on hardware
            
            logger.info(f"Set maximum memory: {max_memory}")
            return max_memory
        else:
            return None

    def generate_from_model(self, inputs: List[str]) -> List[List[str]]:
        """
        queries the generation model for a given batch of inputs

        Returns a list of lists of generated outputs (i.e. batch_size x num_return_sequences)
        """
        encoded_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True)
        # encoded_inputs has shape: [batch_size, seq_len]
        if encoded_inputs['input_ids'].shape[1] > 2048:
            logger.warning(
                f"Encoded inputs sequence length exceeds: {encoded_inputs['input_ids'].shape[1]}." \
                f"This may cause issues...")
        start_time = time.time()
        model_outputs = self.model.generate(
            input_ids=encoded_inputs['input_ids'].cuda(), 
            attention_mask=encoded_inputs['attention_mask'].cuda(),
            max_new_tokens=self.args.max_new_tokens, 
            min_length=self.args.min_length,
            num_beams=self.args.num_beams,
            num_return_sequences=self.args.num_return_sequences, 
            early_stopping=not self.args.no_early_stop,
            do_sample=self.args.do_sample, 
            temperature=self.args.temperature, 
            top_k=self.args.top_k, 
            top_p=self.args.top_p,
            )
        end_time = time.time()

        # model_outputs has shape: [num_return_sequences, seq_len]
        # use the actual batch size instead of args.batch_size as these can differ
        cur_batch_size = encoded_inputs['input_ids'].shape[0]
        new_tokens = (model_outputs.shape[1] - encoded_inputs['input_ids'].shape[1]) * model_outputs.shape[0]
        logger.info(f"Generated {(new_tokens)} new tokens "
                    f"in {end_time - start_time:.4f} seconds "
                    f"(current batch size: {cur_batch_size}).")
        
        model_outputs = self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
        
        return self.reshape_model_outputs(model_outputs, cur_batch_size)

    @staticmethod
    def reshape_model_outputs(outputs: List[str], input_batch_size: int) -> List[List[str]]:
        """
        Reshapes a 1D list of output sequences with size [num_return_sequences]
        to a 2D list of output sequences with size [batch_size, num_return_sequences]
        """
        
        num_return_sequences = len(outputs)
        return_seqs_per_input = num_return_sequences//input_batch_size

        if return_seqs_per_input > 1:
            logger.info(f"Number of return sequences ({num_return_sequences}) > batch size ({input_batch_size})")

        # pack outputs into a list of lists, i.e. batch_size x num_return_seqs
        outputs = [outputs[i:i+return_seqs_per_input]for i in range(0, num_return_sequences, return_seqs_per_input)]
        
        if len(outputs) != input_batch_size:
            raise ValueError(f"Got {len(outputs)} outputs from model but expected {input_batch_size}!")
        
        if len(outputs[0]) != return_seqs_per_input:
            raise ValueError(f"Got {len(outputs[0])} return sequences but expected {return_seqs_per_input}!")

        return outputs

class API_LLM(object):
    """
    Custom class for interacting with the LLMs via APIs.
    """

    def __init__(self, args: InferenceArguments):

        self.args = args

        from langchain.llms import Cohere, OpenAI
        from langchain.chat_models import ChatOpenAI
        
        from api_secrets import COHERE_API_KEY, OPENAI_API_KEY

        if not (self.args.model_name_or_path.lower().startswith("cohere-") or \
            self.args.model_name_or_path.lower().startswith("openai-")):
        
            raise ValueError("Currently only Cohere and OpenAI models are supported! "
                            "Prefix your model name with either 'cohere-' or 'openai-'")
        else:
            provider = self.args.model_name_or_path.lower().split("-")[0]
            # Infer the model name by removing the leading provider name
            model_name = "-".join(self.args.model_name_or_path.lower().split("-")[1:])

            logger.info(f"Loading {provider} model")
            logger.info("Note that full reproducibility is not guaranteed for API models.")

            if provider == "cohere":
                self.model = Cohere(
                    model=model_name,
                    k=self.args.top_k,
                    p=self.args.top_p,
                    max_tokens=self.args.max_new_tokens,
                    frequency_penalty=self.args.frequency_penalty,
                    presence_penalty=self.args.presence_penalty,
                    temperature=self.args.temperature,
                    cohere_api_key=COHERE_API_KEY
                )
            else:
                if self.args.model_name_or_path == 'openai-gpt-3.5-turbo':
                    self.model = ChatOpenAI(
                        model_name=model_name,
                        temperature=self.args.temperature,
                        max_tokens=self.args.max_new_tokens,
                        top_p=self.args.top_p,
                        frequency_penalty=self.args.frequency_penalty,
                        presence_penalty=self.args.presence_penalty,
                        openai_api_key=OPENAI_API_KEY
                        )
                else:
                    # TODO: Consider adjusting parameter `n` (number of generations) or `best_of`
                    self.model = OpenAI(
                        model_name=model_name,
                        temperature=self.args.temperature,
                        max_tokens=self.args.max_new_tokens,
                        top_p=self.args.top_p,
                        frequency_penalty=self.args.frequency_penalty,
                        presence_penalty=self.args.presence_penalty,
                        openai_api_key=OPENAI_API_KEY
                    )

        if self.args.batch_size > 1:
            warnings.warn("Batch size set to value >1. API models only support batch size of 1. "
                          "Will change batch size to 1 automatically.")
            self.args.batch_size = 1
        # track the estimated cost of the API calls
        self.cost = 0
        self.total_tokens = 0

    def generate_from_model(self, inputs: List[str]) -> List[List[str]]:
        """
        queries the generation model for a given batch of inputs

        Returns a list of lists of generated outputs (i.e. batch_size x num_return_sequences)
        """

        from langchain.callbacks import get_openai_callback
        
        start_time = time.time()
        if self.args.model_name_or_path == 'openai-gpt-3.5-turbo':
            # handle chat inputs differently
            from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
            from langchain import LLMChain
            human_template="{text}"
            human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
            chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
            chain = LLMChain(llm=self.model, prompt=chat_prompt)
            with get_openai_callback() as cb:
                outputs = chain.run(inputs[0])
                logger.info(f'{cb}'.replace("\n", ", "))
                self.cost += cb.total_cost
                self.total_tokens += cb.total_tokens
        elif self.args.model_name_or_path.startswith('openai-'):
            with get_openai_callback() as cb:
                outputs = self.model(inputs[0])
                logger.info(f'{cb}'.replace("\n", ", "))
                self.cost += cb.total_cost
                self.total_tokens += cb.total_tokens
        else:
            outputs = self.model(inputs[0])

        end_time = time.time()

        logger.info(f"Processed query in {end_time - start_time:.4f} seconds.")
                
        return [[outputs.strip()]]

# class LLAMA(object):
#     """Wrapper class for LLaMA model. Provides the same interface as the above LLM class for HuggingFace models."""
#     def __init__(self, args: InferenceArguments) -> None:
#         #
#         # from utils.model import setup_model_parallel, load
        
#         self.args = args

#         logger.info("Loading LLaMA model")
#         local_rank, world_size = self.setup_model_parallel(self.args.seed)
#         if local_rank > 0:
#             sys.stdout = open(os.devnull, 'w')

#         # tokenizer is expected to be in the parent directory of the model
#         tokenizer_path = str(Path(self.args.model_name_or_path).parent / "tokenizer.model")
        
#         self.model = self.load(
#             self.args.model_name_or_path, 
#             tokenizer_path, 
#             local_rank, 
#             world_size,
#             max_seq_len=512,
#             max_batch_size=self.args.batch_size
#             )

#         return

#     def generate_from_model(self, inputs: List[str]) -> List[str]:

#         outputs = self.model.generate(
#             inputs, 
#             max_gen_len=self.args.max_new_tokens, 
#             temperature=self.args.temperature, 
#             top_p=self.args.top_p
#         )
#         # pack outputs into a list of lists to match expected format from HF models
#         return [[o] for o in outputs]

#     @staticmethod
#     def setup_model_parallel(seed: int) -> Tuple[int, int]:
#         local_rank = int(os.environ.get("LOCAL_RANK", -1))
#         world_size = int(os.environ.get("WORLD_SIZE", -1))

#         torch.distributed.init_process_group("nccl")
#         initialize_model_parallel(world_size)
#         torch.cuda.set_device(local_rank)

#         # seed must be the same in all processes
#         torch.manual_seed(seed)
#         return local_rank, world_size

#     @staticmethod
#     def load(
#         ckpt_dir: str,
#         tokenizer_path: str,
#         local_rank: int,
#         world_size: int,
#         max_seq_len: int,
#         max_batch_size: int,
#     ) -> LLaMA:
#         start_time = time.time()
#         checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
#         assert world_size == len(
#             checkpoints
#         ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
#         ckpt_path = checkpoints[local_rank]
#         print("Loading")
#         checkpoint = torch.load(ckpt_path, map_location="cpu")
#         with open(Path(ckpt_dir) / "params.json", "r") as f:
#             params = json.loads(f.read())

#         model_args: ModelArgs = ModelArgs(
#             max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
#         )
#         tokenizer = Tokenizer(model_path=tokenizer_path)
#         model_args.vocab_size = tokenizer.n_words
#         torch.set_default_tensor_type(torch.cuda.HalfTensor)
#         model = Transformer(model_args)
#         torch.set_default_tensor_type(torch.FloatTensor)
#         model.load_state_dict(checkpoint, strict=False)

#         generator = LLaMA(model, tokenizer)
#         print(f"Loaded in {time.time() - start_time:.2f} seconds")
#         return generator


if __name__ == "__main__":
    
    hf_parser = HfArgumentParser((InferenceArguments))
    args = hf_parser.parse_args_into_dataclasses()[0]

    llm = LLM(args)
    print(llm.generate_from_model(["This is an awesome prompt :)"]))

    # llm = LLAMA(args)
    # print(llm.generate_from_model(["This is an awesome prompt :)"]))