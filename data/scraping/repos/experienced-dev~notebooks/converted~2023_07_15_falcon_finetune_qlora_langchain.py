#!/usr/bin/env python
# coding: utf-8

# MIT License
# 
# Copyright (c) 2023 Pavel Shibanov [https://blog.experienced.dev/](https://blog.experienced.dev/?utm_source=notebooks)
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


get_ipython().run_line_magic('env', 'PIP_QUIET=1')


# In[ ]:


get_ipython().run_line_magic('pip', 'install      python-dotenv==1.0.0      bitsandbytes==0.40.0      datasets==2.13.1      scipy==1.11.1      einops==0.6.1      xformers==0.0.20      langchain==0.0.234      git+https://github.com/huggingface/transformers.git@91d7df58b6537d385e90578dac40204cb550f706      git+https://github.com/huggingface/accelerate.git@bb47344c774b9508d8171f746632794524160410      git+https://github.com/huggingface/peft.git@06755411549a12da27ba07e4f0ff9699bd6d9194  ')


# In[ ]:


model_name = "tiiuae/falcon-7b-instruct"


# In[ ]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# In[ ]:


import torch
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    load_in_8bit=False,
    device_map="auto",
    trust_remote_code=True,
)
model = base_model


# In[ ]:


from datasets import load_dataset

data = load_dataset("truthful_qa", "generation")
data = data["validation"].filter(lambda item: item["category"] == "Misconceptions")
data


# In[ ]:


import transformers
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain import PromptTemplate, LLMChain
from IPython.display import display, Markdown


default_template = """
Human: {input} 
AI:"""

verbose_template = """
The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: {input} 
AI:"""


def get_chain(model, template, verbose=False):
    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        stop_sequence="\nAI:",
        temperature=0.7,
        max_new_tokens=512,
        repetition_penalty=1.2,
    )
    return LLMChain(
        llm=HuggingFacePipeline(pipeline=pipeline),
        prompt=PromptTemplate.from_template(template),
        verbose=verbose,
    )


base_chain_verbose = get_chain(base_model, verbose_template)
fine_tuned_verbose = get_chain(model, verbose_template)


def compare_results(base_model, fine_tuned_model, item, template=None, verbose=False):
    if template is None:
        template = default_template
    base_chain = get_chain(base_model, template, verbose)
    fine_tuned_chain = get_chain(fine_tuned_model, template, verbose)
    base_res = base_chain.run(item["question"])
    fine_tuned_res = fine_tuned_chain.run(item["question"])
    display(
        Markdown(
            f"""
### question: 
{item['question']}
#### base_model:
{base_res}
#### fine_tuned_model:
{fine_tuned_res}
#### best answer:
{item['best_answer']}
#### source:
{item['source']}
"""
        )
    )


# In[ ]:


compare_results(base_model, model, data[46])


# In[ ]:


import random

compare_results(base_model, model, random.choice(data))


# In[ ]:


compare_results(base_model, model, random.choice(data), verbose_template, verbose=True)


# In[ ]:


fine_tuning_template = """
Human: {question}
AI: {best_answer}
"""
fine_tune_prompt = PromptTemplate.from_template(fine_tuning_template)


def tokenize(item):
    return tokenizer(
        fine_tune_prompt.format(
            question=item["question"], best_answer=item["best_answer"]
        ),
        padding=True,
        truncation=True,
    )


train_dataset = data.map(tokenize)


# In[ ]:


train_dataset


# In[ ]:


import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# In[ ]:


from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True,
)


# In[ ]:


def print_num_params(model):
    params = [
        (param.numel(), param.numel() if param.requires_grad else 0)
        for _, param in model.named_parameters()
    ]
    all, train = map(sum, zip(*params))
    print(f"{train=} / {all=} {train/all:f}")


print_num_params(model)


# In[ ]:


# model.gradient_checkpointing_enable()


# In[ ]:


from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
print_num_params(model)


# In[ ]:


import transformers

output_dir = "fine_tuned"

training_args = transformers.TrainingArguments(
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=1,
    output_dir=output_dir,
    max_steps=100,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()


# In[ ]:


compare_results(base_model, model, data[6])

