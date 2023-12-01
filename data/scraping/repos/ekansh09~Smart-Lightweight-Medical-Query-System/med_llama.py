import os
os.environ["TRANSFORMERS_CACHE"] = "/scratch/megathon/cache/"

import subprocess
import sys
import json
import cleantext
from torch import bfloat16
import transformers
from tqdm import tqdm

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

MODEL_IDS = {
    "alpaca": "medalpaca/medalpaca-7b",
    "meta": "meta-llama/Llama-2-7b-chat-hf",
    "llamaf16": "metaquant.gguf.fp16.bin",
    "llamaq4": "metaquant.gguf.q4_k_m.bin",
    "llamaq5": "metaquant.gguf.q5_k_m.bin",
}


def get_vector_store(kargs, prompt):
    loader = TextLoader(kargs["con_docs"])
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=kargs["chunk_size"], chunk_overlap=kargs["chunk_overlap"])
    all_splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=kargs["embed_file"], 
        model_kwargs={"device": kargs["device"]})
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    return vectorstore

def get_context(vstore, prompt):
    docs = vstore.similarity_search_with_score(prompt)

    context = ""
    for doc in docs:
        context += doc[0].page_content + " "

    new_prompt = f"{context} \n\n Question: {prompt} \n\n Answer:"
    return new_prompt

def quantize_model(model_root, output_name, ggml_version="gguf"):
    """
    Quantizes a model using the llama.cpp script
    model_root: /scratch/megathon/cache/models--meta-llama--Llama-2-7b-chat-hf/snapshots/94b07a6e30c3292b8265ed32ffdeccfdadf434a8
    output_name: /scratch/megathon/quant/metaquant
    """

    fp16_op = f"{output_name}.{ggml_version}.fp16.bin"
    os.system(f"python /scratch/megathon/quant/llama.cpp/convert.py {model_root} --outtype f16 --outfile {fp16_op}")
    
    print("Converted to fp16. Output file: ", fp16_op)
    QUANTIZATION_METHODS = ["q4_k_m", "q5_k_m"]

    for method in QUANTIZATION_METHODS:
        print(f"Quantizing with method {method}")
        qtype = f"{output_name}.{ggml_version}.{method}.bin"
        os.system(f"/scratch/megathon/quant/llama.cpp/quantize {fp16_op} {qtype} {method}")

def make_prediction(model_name, prompt, kargs, ggml_version="gguf", device="cuda"):
    """
    model_name: /scratch/megathon/quant/metaquant
    quant_method: q4_k_m/q5_k_m
    """

    qtype = f"{model_name}.{ggml_version}.{kargs['quant_method']}.bin"
    print(f"Running with quantized model {qtype}")
    # os.system(f"/scratch/megathon/quant/llama.cpp/main -m {qtype} -n {kargs['n']} --log-disable \
    #           --repeat_penalty {kargs['penalty']} --color -ngl {kargs['ngl']} -p \'{prompt}\' ")
    subprocess.call(["/scratch/megathon/quant/llama.cpp/main", "-m", qtype, "-n", str(kargs["n"]), "--log-disable", 
        "--repeat_penalty", str(kargs["penalty"]), "--color", "-ngl", str(kargs["ngl"]), "-p", f'\"{prompt}\"', "|", "output.txt"])
    with open('output.txt') as f:
        lines = f.readlines()
    os.remove("output.txt")
    return lines

def load_model(model_name, device="cuda"):
    model_id = MODEL_IDS[model_name]
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    hf_auth = "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"
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        token=hf_auth,
        device_map=device,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map=device,
        token=hf_auth
    )

    model.eval()
    print(f"Model loaded on {device}")

def generate_context_docs(json_path, output_path="webmd_context_docs.txt"):
    """
    Generates a file with all the context fields from the json file
    """

    with open(json_path) as f:
        data = json.load(f)

    with open(output_path, "a") as f:
        for x in range(len(data["data"])):
            inp = data["data"][x]["paragraphs"][0]["context"]
            inp = cleantext.clean(inp, clean_all=False, extra_spaces=True, stemming=False, stopwords=False,
                lowercase=False, numbers=False, punct=False)
            
            # remove some non info lines
            if "var s_context" in inp:
                continue
            f.write(inp)
            f.write("\n\n")


class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )

        self.outnull_file.close()
        self.errnull_file.close()


if __name__ == "__main__":
    generate_context_docs("/home/abhiroop.talasila/megathon/data 2/train_webmd_squad_v2_full.json")
    generate_context_docs("/home/abhiroop.talasila/megathon/data 2/val_webmd_squad_v2_consec.json")
    generate_context_docs("/home/abhiroop.talasila/megathon/data 2/val_webmd_squad_v2_full.json")