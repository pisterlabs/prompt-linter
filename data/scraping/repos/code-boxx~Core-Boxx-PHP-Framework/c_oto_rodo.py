# AUTO LOADER
# credits - some parts "borrowed" from oobabooga
# https://github.com/oobabooga/text-generation-webui/blob/main/modules/models.py

# (A) LOAD SETTINGS
import a_settings as set

# (B) MANUALLY SPECIFIED MODEL - USE LLAMA CPP
if hasattr(set, "model_file"):
  from langchain.llms import LlamaCpp
  llm = LlamaCpp(
    model_path = set.model_file,
    ** set.model_args
  )

# (C) HUGGING FACE
else:
  # (C1) IMPORT TRANSFORMERS MODULES
  import torch, psutil
  from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
  from accelerate import infer_auto_device_map, init_empty_weights
  from langchain import HuggingFacePipeline

  # (C2) HELPER - AUTO MAX MEMORY CALCULATION
  def max_mem():
    # (C2-1) GPU MEMORY
    total = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
    suggestion = round((total - 1000) / 1000) * 1000
    if total - suggestion < 800:
      suggestion -= 1000
    suggestion = int(round(suggestion / 1000))
    max = { 0 : f"{suggestion}GiB" }

    # (C2-2) CPU MEMORY
    total = (psutil.virtual_memory().available / (1024 * 1024))
    suggestion = round((total - 1000) / 1000) * 1000
    if total - suggestion < 800:
      suggestion -= 1000
    suggestion = int(round(suggestion / 1000))
    max["cpu"] = f"{suggestion}GiB"

    # (C2-3) RETURN CALCULATED MEMORY
    return max

  # (C3) INIT MODEL PARAMS
  params = {
    "device_map" : "auto"
  }

  # (C4) GPU ACCELERATED
  if set.gpu:
    config = AutoConfig.from_pretrained(set.model_name)
    with init_empty_weights():
      model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()
    params["device_map"] = infer_auto_device_map(
      model,
      dtype = config.torch_dtype,
      max_memory = max_mem(),
      no_split_module_classes = model._no_split_modules
    )

  # (C5) CPU ONLY
  else:
    params["torch_dtype"] = torch.float32

  # (C6) LOAD MODEL
  model = AutoModelForCausalLM.from_pretrained(set.model_name, **params)

  # (C7) LLM/PIPE
  llm = HuggingFacePipeline(pipeline = pipeline(
    task = "text-generation",
    model = model,
    tokenizer = AutoTokenizer.from_pretrained(set.model_name),
    ** set.model_args
  ))