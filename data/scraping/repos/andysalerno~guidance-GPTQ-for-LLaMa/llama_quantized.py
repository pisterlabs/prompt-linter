from pathlib import Path
from guidance.llms._transformers import Transformers
from guidance.llms._llm import LLM
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import transformers
from utils import find_layers
import quant


class LLaMAQuantized(Transformers):
    """ A HuggingFace transformers version of the LLaMA language model with Guidance support.
    """

    cache = LLM._open_cache("_llama.diskcache")

    def __init__(self, model_dir, model, tokenizer=None, device_map=None, wbits=4, groupsize=128, **kwargs):
        """ Create a new LLaMA model.
        """

        # load the LLaMA specific tokenizer and model
        if isinstance(model, str):
            model_name = model
            model = load_quantized(model, wbits, groupsize, model_dir)

            tokenizer_path = f'{model_dir}/{model_name}/'
            print(f'Loading tokenizer from: {tokenizer_path}')

            tokenizer = LlamaTokenizer.from_pretrained(Path(tokenizer_path), clean_up_tokenization_spaces=True)

        super().__init__(model, tokenizer=tokenizer, device_map=device_map, **kwargs)


    @staticmethod
    def role_start(role):
        if role == 'user':
            return 'USER: '
        elif role == 'assistant':
            return 'ASSISTANT: '
        else:
            return ''
    
    @staticmethod
    def role_end(role):
        if role == 'user':
            return ''
        elif role == 'assistant':
            return '</s>'
        else:
            return ''

def load_quantized(model_name, wbits, groupsize, model_dir):
    # Find the quantized model weights file (.pt/.safetensors)
    path_to_model = Path(f'{model_dir}/{model_name}')
    pt_path = find_quantized_model_file(model_dir, model_name, wbits, groupsize)
    if not pt_path:
        exit()
    else:
        print(f"Found the following quantized model: {pt_path}")

    # qwopqwop200's offload
    model = _load_quant(str(path_to_model), str(
        pt_path), wbits, groupsize)

    # No offload
    print('Model to device')
    model = model.to(torch.device('cuda:0'))

    return model


def _load_quant(model, checkpoint, wbits, groupsize=-1, exclude_layers=None, eval=True):
    exclude_layers = exclude_layers or ['lm_head']

    def noop(*args, **kwargs):
        pass

    config = AutoConfig.from_pretrained(
        model, trust_remote_code=False)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=False)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()

    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]

    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    model.seqlen = 2048

    return model


# Used to locate the .pt/.safetensors quantized file
def find_quantized_model_file(model_dir, model_name, wbits, groupsize):
    path_to_model = Path(f'{model_dir}/{model_name}')
    pt_path = None
    priority_name_list = [
        Path(
            f'{model_dir}/{model_name}{hyphen}{wbits}bit{group}{ext}')
        for group in ([f'-{groupsize}g', ''] if groupsize > 0 else [''])
        for ext in ['.safetensors', '.pt']
        for hyphen in ['-', f'/{model_name}-', '/']
    ]

    for path in priority_name_list:
        if path.exists():
            pt_path = path
            break

    # If the model hasn't been found with a well-behaved name, pick the last .pt
    # or the last .safetensors found in its folder as a last resort
    if not pt_path:
        found_pts = list(path_to_model.glob("*.pt"))
        found_safetensors = list(path_to_model.glob("*.safetensors"))
        pt_path = None

        if len(found_pts) > 0:
            if len(found_pts) > 1:
                print(
                    'More than one .pt model has been found. The last one will be selected. It could be wrong.')

            pt_path = found_pts[-1]
        elif len(found_safetensors) > 0:
            if len(found_pts) > 1:
                print(
                    'More than one .safetensors model has been found. The last one will be selected. It could be wrong.')

            pt_path = found_safetensors[-1]

    return pt_path
