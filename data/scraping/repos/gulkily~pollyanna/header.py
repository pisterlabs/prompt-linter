import pickle
import threading
import time

import fire as fire
from tiktoken import get_encoding

import openai


def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        wrapper.elapsed_time_per_thread[threading.get_ident()] = end - start

        def _get_elapsed_time():
            return wrapper.elapsed_time_per_thread[threading.get_ident()]

        wrapper.get_elapsed_time = _get_elapsed_time
        return result

    wrapper.elapsed_time_per_thread = {}
    return wrapper


def get_tokenized_length(s: str):
    return len(get_encoding("p50k_base").encode(s))


class PromptRunner:
    num_calls_lock = threading.Lock()
    num_calls = 0
    sum_prompt_tokens = 0
    sum_completion_tokens = 0

    read = True  # should I read from cache saved prompt completions?
    first = True  # is this the first time I'm entering, if so, need to initialize / read if read is True.
    prompts_to_completions = {}  # the cache
    used_saved_prompts = 0

    def __init__(self, name):
        self.name = name
        # name of the cached pkl to load prompts_to_completions from
        self.pkl = f"mem_prompt_and_comp__{name}.pkl"
        self.get_elapsed_time = None

    def main_body(self, model_name, prompt, do_print):
        if do_print:
            print("####################################################")
            print("[run_prompt] print prompt")
            print(prompt)
            print("[run_prompt] done print prompt")
        self.num_calls += 1

        def get_stat_str():
            return f"{self.num_calls=}, {self.sum_prompt_tokens=}, {self.sum_completion_tokens=}; " \
                   f"{self.used_saved_prompts=}; {len(self.prompts_to_completions)=}"

        # breakpoint()
        if prompt in self.prompts_to_completions:
            self.used_saved_prompts += 1
            print(f"[run_prompt] used saved prompts: # {self.used_saved_prompts=}")
            completion = self.prompts_to_completions[prompt]
        else:
            print(f"[run_prompt] running openai.ChatCompletion.create("
                  f"prompt s.t. {len(prompt)=}, model='{model_name}'); {get_stat_str()}")
            # if len(prompt) >= 6000:
            #     breakpoint()  # todo: you sure?
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.88
            )["choices"][0]["message"]["content"]

            assert prompt not in self.prompts_to_completions
            self.prompts_to_completions[prompt] = completion

        self.sum_completion_tokens += get_tokenized_length(completion)

        with open(self.pkl, 'wb') as f:  # open a pkl file
            pickle.dump(self.prompts_to_completions, f)  # serialize the dict

        if do_print:
            print("[run_prompt] print completion")
            print(completion)
            print(
                f"[run_prompt] done print completion;  s.t. {len(prompt)=}, model='{model_name}'); {get_stat_str()}")
        print(f"[run_prompt] done")
        print("####################################################")

        return completion

    @measure_time
    def run_prompt(self, prompt: str, model: str = "gpt-4", do_print=True) -> str:
        if self.read and self.first:
            self.first = False
            try:
                with open(self.pkl, 'rb') as f:
                    self.prompts_to_completions = pickle.load(f)  # deserialize using load()
            except Exception as _:
                self.prompts_to_completions = {}

        self.num_calls_lock.acquire()
        self.sum_prompt_tokens += get_tokenized_length(prompt)
        self.num_calls_lock.release()

        assert isinstance(model, str)
        model_name = model

        max_num_attempts = 3
        for attempt_id in range(max_num_attempts):
            try:
                return self.main_body(model_name, prompt, do_print)
            except Exception as e:
                print("[run_prompt] Failed with error:", str(e))
                print("[run_prompt] trying again, attempt", attempt_id + 1)
        assert False, f"Failed to get a completion in {max_num_attempts} attempts."


def run_prompt(file_path):
    with open(file_path, "r") as file:
        txt = file.read()
    PromptRunner("[run_prompt]").run_prompt(txt)


if __name__ == "__main__":
    fire.Fire(run_prompt)
