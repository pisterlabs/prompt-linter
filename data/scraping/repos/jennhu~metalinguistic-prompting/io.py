import json
from time import gmtime, strftime
import argparse
import torch

from . import openai_api, models

# Extremely basic helper functions.
def timestamp():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def dict2json(d, out_file):
    with open(out_file, "w") as fp:
        json.dump(d, fp, indent=2)

def json2dict(in_file):
    with open(in_file, "r") as fp:
        d = json.load(fp)
    return d

# Helper function for parsing command-line arguments.
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model using specified prompts")
    parser.add_argument("--model", "-M", type=str, help="Name of model")
    parser.add_argument("--model_type", type=str, choices=["openai", "hf"])
    parser.add_argument("--key", "-K", type=str, default="key.txt", 
                        help="Path to file with secret OpenAI API key")
    parser.add_argument("--seed", "-S", type=int, default=0, 
                        help="Random seed for reproducibility")
    parser.add_argument("--eval_type", type=str, default="direct", 
                        choices=[
                            "direct", 
                            "metaQuestionSimple", 
                            "metaInstruct", 
                            "metaQuestionComplex"
                        ], 
                        help="Type of evaluation (for prompt design)")
    parser.add_argument("--option_order", type=str, default="goodFirst",
                        choices=["goodFirst", "badFirst"]),
    parser.add_argument("--data_file", type=str,
                        help="Path to data containing prefixes for next-word prediction task")
    parser.add_argument("--out_file", type=str,
                        help="Path to save output JSON file")
    parser.add_argument("--dist_folder", type=str, default=None,
                        help="(OPTIONAL) path to folder to save distribution files (as .npy)")
    args = parser.parse_args()
    return args

# Helper function for initializing models.
def initialize_model(args):
    # Set device to GPU if cuda is available.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Set device to CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU (CUDA unvailable); adjust your expectations")
        
    # Initialize model based on model type and name.
    if args.model_type == "openai":
        # Secret file with API key (DO NOT commit this)
        openai_api.set_key_from_file(args.key)
        model = models.OpenAI_LLM(args.eval_type, args.model, args.seed)
    else:
        if "flan-t5" in args.model:
            model = models.T5_LLM(args.eval_type, args.model, args.seed, device=device)
        else:
            raise ValueError(
                f"Model not supported! (Your model: {args.model})"
            )
    return model