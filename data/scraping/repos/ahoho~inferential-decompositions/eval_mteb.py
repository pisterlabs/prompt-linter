
import argparse
import json
import warnings
import logging
from typing import Optional, Union
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer
import openai
import langchain
from langchain.cache import SQLiteCache


from config import load_config
from generation_utils import (
    OpenAIPB,
    OpenAIChatPB,
    HuggingFacePipelineDS,
    load_hf_generation_pipeline,
    load_adapted_hf_generation_pipeline,
)

logger = logging.getLogger("mteb.evaluation.MTEB")

warnings.filterwarnings(
    "ignore", 
    message="Both `max_new_tokens`.*",
)

def write_jsonl(data, fpath):
    with open(fpath, "w") as outfile:
        for index, line in enumerate(data): 
            s = json.dumps(line, ensure_ascii=False)
            if index == len(data) - 1:
                outfile.write(s)
            else: 
                outfile.write(f"{s}\n")


def read_jsonl(fpath):
    with open(fpath) as infile:
        return [json.loads(line) for line in infile if line]


class GenerationEmbedder():
    def __init__(
        self,
        instructions: str,
        openai_api_key: str,
        exemplar_pool: Optional[list[str]] = None,
        exemplar_format: str = "{input}->{output}",
        exemplar_sep: str = "\n",
        multi_output_sep: Optional[str] = " | ",
        exemplars_per_prompt: Optional[int] = None,
        draws_per_pool: int = 1,
        repeat_draws: bool = False,
        shuffles_per_draw: int = 1,
        output_combination_strategy: str = "concatenate_text",
        include_original_doc: bool = True,
        embedding_model_name: str = "all-mpnet-base-v2",
        gen_model_name: str = "text-curie-001",
        generations_per_prompt: int = 1,
        temperature: float = 0,
        top_p: float = 1.,
        generation_kwargs: Optional[dict] = None,
        max_tokens: int = 50,
        cache_db_path: str = ".langchain.db",
        dry_run: bool = False,
        device: str = "cpu",
        seed: Optional[int] = None,

    ) -> None:

        self.cache_db_path = cache_db_path
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)

        # if we are using an adapter model with PEFT, then pass a list with the base
        # model and gen model
        lora_model_name = None
        if isinstance(gen_model_name, list):
            gen_model_name, lora_model_name = gen_model_name


        generation_kwargs = generation_kwargs if generation_kwargs is not None else {}
        if "gpt-3.5-turbo" in gen_model_name:
            if self.cache_db_path is not None:
                logging.warning("Disabling cache for gpt-3.5-turbo. This is a known bug.")
                self.cache_db_path = None

            self._chat_prefix_messages = [
                {"role": "system", "content": "You are a helpful and intelligent assistant."},
            ] # add during prompting
            self.llm = OpenAIChatPB(
                model_name=gen_model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                openai_api_key=openai_api_key,
                **generation_kwargs,
            )
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.max_model_tokens = self.llm.modelname_to_contextsize(gen_model_name)

        elif gen_model_name.startswith("text-"):
            self.llm = OpenAIPB(
                model_name=gen_model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                openai_api_key=openai_api_key,
                n=generations_per_prompt,
                best_of=generations_per_prompt,
                **generation_kwargs,
            )
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.max_model_tokens = self.llm.modelname_to_contextsize(gen_model_name)
            generations_per_prompt = 1 # handled by API
        elif lora_model_name is None:
            logging.info(f"Assuming that {gen_model_name} is a hugginface model")
            pipe = load_hf_generation_pipeline(
                gen_model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                device=device,
                generation_kwargs=generation_kwargs,
            )
            # TODO: support `generations_per_prompt`
            self.llm = HuggingFacePipelineDS(pipeline=pipe)
            self.tokenizer = pipe.tokenizer
            self.max_model_tokens = self.tokenizer.model_max_length
        else:
            logging.info(
                f"Assuming that {gen_model_name} is a hugginface model and "
                f"{lora_model_name} is the adapter."
            )
            pipe = load_adapted_hf_generation_pipeline(
                gen_model_name,
                lora_model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                device=device,
                generation_kwargs=generation_kwargs,
            )
            # TODO: support `generations_per_prompt`
            self.llm = HuggingFacePipelineDS(pipeline=pipe)
            self.tokenizer = pipe.tokenizer
            self.max_model_tokens = self.tokenizer.model_max_length
        
        self.max_gen_tokens = max_tokens
        
        logger.info(f"cacke db path is {self.cache_db_path}")
        if self.cache_db_path is not None:
            langchain.llm_cache = SQLiteCache(database_path=self.cache_db_path)

        self.instructions = instructions

        self.gen_model_name = gen_model_name

        self.exemplar_pool = exemplar_pool
        self.exemplars_per_prompt = exemplars_per_prompt

        self.draws_per_pool = draws_per_pool
        self.repeat_draws = repeat_draws
        self.shuffles_per_draw = shuffles_per_draw
        self.generations_per_prompt = generations_per_prompt

        self.exemplar_format = exemplar_format
        self.multi_output_sep = multi_output_sep
        self.exemplar_sep = exemplar_sep

        self.include_original_doc = include_original_doc
        self.dry_run = dry_run
        self.seed = seed
        self.random_state = np.random.default_rng(seed)
        self.torch_random_state = torch.manual_seed(seed)

        if output_combination_strategy not in ["concat_strings", "mean_embeds", "concat_embeds", "concat_doc_gen", "list_embeds"]:
            raise NotImplementedError(f"`output_combination_strategy` {output_combination_strategy} not supported")
        if output_combination_strategy == "concat_doc_gen" and not include_original_doc:
            raise ValueError("Have to set `include_original_doc=True` for `concat_doc_gen` strategy")
        self.output_combination_strategy = output_combination_strategy
        self._truncation_counter = 0
        self._generated = {}
        self._set_num_prompt_tokens()

    
    def _reset_seed(self):
        self.random_state = np.random.default_rng(self.seed)
        self.torch_random_state = torch.manual_seed(self.seed)


    def get_num_tokens(self, text):
        return len(self.tokenizer.encode(text, truncation=False, add_special_tokens=False))
    

    def _set_num_prompt_tokens(self):
        """Get number of tokens in input and output prefixes"""
        exemplar_format = self.exemplar_format.format(input="", output="")
        prompt_tokens = self.get_num_tokens(exemplar_format)
        if self.instructions:
            prompt_tokens += self.get_num_tokens(self.instructions)
        self._prompt_tokens = prompt_tokens


    def _truncate_text(self, input, est_chars_per_tok=3.5): # conservative estimate
        """Make sure text and output fits in model context window"""
        # TODO: replace with tiktoken
        max_model_tokens = self.max_model_tokens # these are tokens in the model context
        max_output_tokens = self.max_gen_tokens # these are tokens to be *generated*
        max_input_tokens = max_model_tokens - max_output_tokens - self._prompt_tokens

        # shortcut: if it looks like we're roughly within the limit, then don't tokenize
        if len(input) / est_chars_per_tok < max_input_tokens:
            return input
        
        # tokenize the text and get the truncated input
        tokenized = self.tokenizer.encode(
            input, max_length=max_input_tokens, truncation=True, add_special_tokens=False
        )

        # check with actual tokenization
        if len(tokenized) < max_input_tokens:
            return input
        
        # if too short, then truncate the input
        self._truncation_counter += 1
        truncated_input = self.tokenizer.decode(tokenized)
        # end on a sentence; otherwise, end on word
        try:
            last_idx = input.rindex(".", 0, len(truncated_input))
        except ValueError:
            last_idx = input.rindex(" ", 0, len(truncated_input))
        return input[:last_idx].strip() + " [...]"


    def sample_exemplars(
        self,   
    ) -> list:
        """
        There can be two sources of randomization for each query to the LLM
        (1) the exemplars shown to the model (a "draw") and
        (2) the order of those exemplars (a "shuffle")

        Ported over from another project based around classification, so not really
        necessary for generation use case
        """
        if not self.exemplars_per_prompt:
            yield None
            return
        self._reset_seed()
        pool_size = len(self.exemplar_pool)
        self.random_state.shuffle(self.exemplar_pool)

        n_exemplars = min(pool_size, self.exemplars_per_prompt)
        n_draws = min(self.draws_per_pool, pool_size // n_exemplars + 1)
        for draw in range(n_draws):
            if self.repeat_draws: # re-sample exemplars
                exemplars = self.random_state.choice(self.exemplar_pool, size=n_exemplars, repeat=False)
            else: # iterate through pool (recall: it is already shuffled)
                start, end = draw*n_exemplars, (1 + draw)*n_exemplars
                exemplars = self.exemplar_pool[start:end]
            if len(exemplars):
                for shuffle in range(self.shuffles_per_draw):
                    self.random_state.shuffle(exemplars)
                    yield exemplars
    

    def create_prompt(
        self,
        instance: str,
        exemplars: Optional[list[str]] = None,
        replace_newlines: bool = True,
    ) -> str:
        if isinstance(self.llm, langchain.llms.OpenAIChat):
            return self._create_chat_prompt(instance, exemplars, replace_newlines)
        else:
            return self._create_prompt(instance, exemplars, replace_newlines)


    def _create_prompt(
        self,
        instance: str,
        exemplars: Optional[list[str]] = None,
        replace_newlines: bool = True,
    ) -> str:
        """
        Generate a prompt for the LLM
            `instance`: the instance to be classified,
            `exemplars`: a list of (input, output) pairs
            `replace_newlines`: replace newlines with spaces
        """
        # TODO: use langchain?
        prompt = ""
        instance_tokens = self.get_num_tokens(instance)
        max_input_tokens = self.max_model_tokens - self.max_gen_tokens - self._prompt_tokens
        remaining_tokens = max_input_tokens - instance_tokens

        if self.instructions is not None:
            prompt += self.instructions

        if exemplars is not None and remaining_tokens > 0:
            for input, output in exemplars:
                if isinstance(output, (list, tuple)):
                    output = self.multi_output_sep.join(output)
                if replace_newlines:
                    input = input.replace("\n", " ")
                    output = output.replace("\n", " ")
                # don't add exemplars if prompt will be too long
                exemplar_formatted = self.exemplar_format.format(input=input, output=output) + self.exemplar_sep
                exemplar_tokens = self.get_num_tokens(exemplar_formatted)
                if remaining_tokens - exemplar_tokens > 0:
                    prompt += exemplar_formatted
                    remaining_tokens -= exemplar_tokens
                else:
                    break
        
        if replace_newlines:
            instance = instance.replace("\n", " ")

        if remaining_tokens < 0:
            instance = self._truncate_text(instance) # takes into account instruction & exemplar format
        
        # we want to set off the trigger for the instance correctly, e.g.,
        #    exemplar_format = "Q: {input} A: {output}"
        # => instance_format = "Q: {input} A: "
        instance_format = self.exemplar_format.split("{output}")[0]
        prompt += instance_format.format(input=instance)        
        return prompt #.strip(" ") # LLM prefers no trailing whitespace TODO: is true?


    def _create_chat_prompt(
        self,
        instance: str,
        exemplars: Optional[list[str]] = None,
        replace_newlines: bool = False,
    ) -> str:
        """
        Generate a prompt for ChatGPT APi
            `instance`: the instance to be classified,
            `exemplars`: a list of (input, output) pairs
            `replace_newlines`: replace newlines with spaces
        """
        messages = self._chat_prefix_messages.copy()
        instance_tokens = self.get_num_tokens(instance)
        max_input_tokens = self.max_model_tokens - self.max_gen_tokens - self._prompt_tokens
        remaining_tokens = max_input_tokens - instance_tokens

        if self.instructions is not None:
            messages.append({"role": "user", "content": self.instructions})
    
        if exemplars is not None and remaining_tokens > 0:
            for i, (input, output) in enumerate(exemplars):
                if isinstance(output, (list, tuple)):
                    output = self.multi_output_sep.join(output)
                if replace_newlines:
                    input = input.replace("\n", " ")
                    output = output.replace("\n", " ")
                # don't add exemplars if prompt will be too long
                exemplar_tokens = self.get_num_tokens(input+"\n"+output)
                if remaining_tokens - exemplar_tokens > 0:
                    if i == 0 and self.instructions is not None:
                        messages[-1]["content"] += input # add to instruction
                    else:
                        messages.append({"role": "user", "content": input})
                    messages.append({"role": "assistant", "content": output})
                    remaining_tokens -= exemplar_tokens
                else:
                    break
        
        
        if replace_newlines:
            instance = instance.replace("\n", " ")

        if remaining_tokens < 0:
            instance = self._truncate_text(instance) # takes into account instruction & exemplar format
        
        return messages, instance


    def process_completions(
        self,
        completion_set: Union[list[LLMResult], LLMResult],
    ) -> dict:
        """
        Consolidate different LM outputs for the same set of inputs.

        If `completion_set`, it consists of different `LLMResult`s for the same
        sequence of input instances. Each `LLMResult` is basically a list of outputs.
        """
        if not isinstance(completion_set, list):
            completion_set = [completion_set]
        
        if isinstance(self.llm,  langchain.llms.OpenAIChat): 
            n_outputs = len(completion_set[0])
            consolidated = [[] for _ in range(n_outputs)]
            for outputs in completion_set: 
                for i, gen in enumerate([x.generations[0][0].text for x in outputs]):         
                    texts = gen.strip().split(self.multi_output_sep)
                    consolidated[i].extend([y.strip() for y in texts])

        else:
            n_outputs = len(completion_set[0].generations)
            consolidated = [[] for _ in range(n_outputs)]
            for outputs in completion_set:
                # for each set, unpack the completed text for each input
                for i, gens in enumerate(outputs.generations):
                    for gen in gens:
                        texts = gen.text.strip().split(self.multi_output_sep)
                        consolidated[i].extend(texts)


        return consolidated


    def generate_from_inputs(
        self,
        inputs: Union[str, list[str]],
    ) -> list[str]:
        """
        Generate completions for each input
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        completions_per_draw = []
        prompts_per_draw = []
        for exemplars in self.sample_exemplars():
            prompts = [self.create_prompt(instance, exemplars) for instance in inputs]
            if self.dry_run:
                logger.warning("\n<<<<< prompt >>>>>\n".join(prompts))
                continue
            # this will be 1 for standard OpenAI chain, since it is handled by the API
            for i in range(self.generations_per_prompt):
                # `generate` does caching; we overload stop tokens to not rely on the cache
                completions = self.llm.generate(prompts, stop=[self.exemplar_sep, f"__index{i}__"])
                completions_per_draw.append(completions)
            prompts_per_draw.append(prompts)

        logging.warning(f"Truncated {self._truncation_counter} out of {len(prompts)*len(prompts_per_draw)} prompts")
        if self.dry_run:
            quit()
        return self.process_completions(completions_per_draw)

    
    def encode(self, sentences, generations=None, topics=None, **embed_kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            generations (`List[List[str]]`): List of existing generations for each sentence
            batch_size (`int`): Batch size for the encoding
            show_progress_bar (`bool`): Show a progress bar or not

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """

        embed_kwargs["show_progress_bar"] = True
        sentences_with_topics = sentences
        if topics is not None:
            assert(len(topics) == len(sentences))
            sentences_with_topics = [f"[Topic: {topic.lower()}] {sent}" for topic, sent in zip(topics, sentences)]
        if generations is None:
            generations = self.generate_from_inputs(sentences_with_topics)
            self._generated.update({s: g for s, g in zip(sentences, generations)})

        if self.include_original_doc and not self.output_combination_strategy == "concat_doc_gen":
            generations = [[s, *gens] for s, gens in zip(sentences, generations)]
        # TODO: do uniqueness filtering here
        if self.output_combination_strategy == "concat_strings":
            generations = [self.multi_output_sep.join(set(gens)) for gens in generations]
            embeddings = self.embedding_model.encode(generations, **embed_kwargs)
        elif self.output_combination_strategy in ["mean_embeds", "concat_doc_gen"]:
            # unpack generations, encode, then take mean per example
            idxs = np.array([i for i, gens in enumerate(generations) for _ in gens])
            gens_unpacked = [g for gens in generations for g in gens]
            embeddings_unpacked = self.embedding_model.encode(gens_unpacked, **embed_kwargs)
            embeddings = np.zeros((len(generations), embeddings_unpacked.shape[1]), dtype=embeddings_unpacked.dtype)
            for i in range(len(generations)): # TODO a random baseline
                embeddings[i] = embeddings_unpacked[idxs==i].mean(0)

            # for concat_doc_gen, we concatenate the original doc with the mean of the generations
            if self.output_combination_strategy == "concat_doc_gen":
                doc_embeddings = self.embedding_model.encode(sentences, **embed_kwargs)
                embeddings = np.hstack([doc_embeddings, embeddings])
        elif self.output_combination_strategy == "concat_embeds":
            # concatenate embeddings for the generated outputs
            # there can be different numbers of generations per example; we calculate
            # the mean. then, for each example, we select that mean number of generations
            # and repeat if necessary (sort of silly; could also take the min)
            mean_n = int(np.mean([len(g) for g in generations]))
            embeddings = np.hstack([
                self.embedding_model.encode([
                        gens[i % len(gens)] for gens in generations
                    ],
                    **embed_kwargs
                ) for i in range(mean_n)
            ])
        elif self.output_combination_strategy == "list_embeds":
            idxs = np.array([i for i, gens in enumerate(generations) for _ in gens])
            gens_unpacked = [g for gens in generations for g in gens]
            embeddings_unpacked = self.embedding_model.encode(gens_unpacked, **embed_kwargs)
            embeddings = [embeddings_unpacked[idxs==i] for i in range(len(generations))]
        else:
            raise NotImplementedError(f"`output_combination_strategy` {self.output_combination_strategy} not supported")
        return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user_config_file", default=None, nargs="?")
    parser.add_argument("--base_config_file", default=None)
    parser.add_argument("--collect_results_dir", default=None)
    args = parser.parse_args()
    config = load_config(args.user_config_file, args.base_config_file)
    if "openai_model_name" in config["llm"]:
        config["llm"]["gen_model_name"] = config["llm"]["openai_model_name"]
    if config["embeddings"]["do_gen"]:
        # load exemplars
        exemplar_pool = None
        if config["exemplars"]["exemplars_per_prompt"] > 0: # don't bother loading if we're not using them
            if config["data"]["exemplar_data_sources"]:
                exemplar_pool = create_exemplar_pool(
                    config["data"]["exemplar_data_sources"],
                    exemplar_dir=config["data"]["exemplars_path"],
                )
            elif config["data"]["exemplars_path"] and Path(config["data"]["exemplars_path"]).is_file():
                exemplar_pool = read_jsonl(config["data"]["exemplars_path"])

        # build generation embedder
        model = GenerationEmbedder(
            instructions=config["data"]["instructions"],
            openai_api_key=config["llm"]["openai_api_key"],
            exemplar_pool=exemplar_pool,
            exemplar_format=config["exemplars"]["format"],
            exemplar_sep=config["exemplars"]["separator"],
            multi_output_sep=config["exemplars"]["multi_output_separator"],
            exemplars_per_prompt=config["exemplars"]["exemplars_per_prompt"],
            draws_per_pool=config["exemplars"]["draws_per_pool"],
            repeat_draws=config["exemplars"]["repeat_draws"],
            shuffles_per_draw=config["exemplars"]["shuffles_per_draw"],
            output_combination_strategy=config["embeddings"]["output_combination_strategy"],
            include_original_doc=config["embeddings"]["include_original_doc"],
            embedding_model_name=config["embeddings"]["embedding_model_name"],
            gen_model_name=config["llm"]["gen_model_name"],
            generations_per_prompt=config["llm"]["generations_per_prompt"],
            temperature=config["llm"]["temperature"],
            top_p=config["llm"]["top_p"],
            generation_kwargs=config["llm"]["generation_kwargs"],
            max_tokens=config["llm"]["max_tokens"],
            cache_db_path=config["main"]["cache_db_path"],
            dry_run=config["main"]["dry_run"],
            device=config["embeddings"]["device"],
            seed=config["main"]["seed"],
        )
    else:
        model = SentenceTransformer(
            config["embeddings"]["embedding_model_name"],
            device=config["embeddings"]["device"],
        )
    evaluation = MTEB(
        tasks=config["data"]["eval_task_names"],
        task_langs=config["data"]["task_langs"],
    )
    evaluation.run(
        model,
        output_folder=config["main"]["results_dir"],
        eval_splits=config["data"]["eval_splits"],
        limit=config["data"]["subsample_size"],
        verbosity=2
    )
    config["llm"].pop("openai_api_key")

    for name in config["data"]["eval_task_names"]:
        result_fpath = Path(config["main"]["results_dir"], f"{name}.json")
        if not result_fpath.exists():
            continue
            
        result_data = json.loads(result_fpath.read_text())
        result_data["config"] = config
        result_fpath.write_text(json.dumps(result_data, indent=2))

        generation_fpath = Path(config["main"]["results_dir"], f"{name}_generations.json")
        if generation_fpath.exists():
            generation_data = json.loads(generation_fpath.read_text())
            generation_data["experiment_name"] = config["main"]["experiment_name"]
            generation_data["experiment_date"] = config["main"]["date"]
            generation_fpath.write_text(json.dumps(generation_data, indent=2))
    if args.collect_results_dir:
        df = collect_results(args.collect_results_dir)
        save_results(df, Path(args.collect_results_dir, "results.csv"))