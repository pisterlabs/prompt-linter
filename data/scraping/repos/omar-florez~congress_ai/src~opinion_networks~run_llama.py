import os, sys
sys.path.append(".")
from tqdm import tqdm
import pdb

from opinion_networks.dataset import Summary, LawDataset
from opinion_networks.nn import MLP
from opinion_networks import trace_graph

import torch
from torch import cuda, bfloat16
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
import random

def background_fn():
    cities = [
        "Lima",
        "Arequipa",
        "Cusco",
        "Trujillo",
        "Chiclayo",
        "Puno",
        "Iquitos",
        "Cajamarca",
        "Tacna",
        "Huancayo",
        "Piura",
        "Ayacucho",
        "Chimbote",
        "Huaraz",
        "Tumbes",
        "Puerto Maldonado"
    ]
    background = f"A person from {random.choice(cities)} in Peru."
    return background

def load_llm(model_id, hf_auth):
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )
    model.eval()
    print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    stop_list = ['\nHuman:', '\n```\n']
    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

    # define custom stopping criteria object
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        #stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=1000,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )


    llm = HuggingFacePipeline(pipeline=generate_text)
    return (model, llm)

def run(model_id, openai_auth, hf_auth):    
    os.environ["SERPAPI_API_KEY"] = hf_auth
    os.environ["OPENAI_API_KEY"] = openai_auth
    
    # Info: https://huggingface.co/docs/hub/security-tokens
    model, llm = load_llm(model_id, hf_auth)
    raw_text_root = "data/peru/laws/texts"
    crawled_files_root = 'data/peru/laws/crawled/'
    summaries_root = 'data/peru/laws/summaries'
    dataset = LawDataset(raw_text_root, crawled_files_root, summaries_root, llm=llm)
    x, y = dataset.load()

    # TODO: test model = MLP(1, [1, 1]), model = MLP(1, [1])
    #model = MLP(1, [3, 1])
    model = MLP(1, [2, 1], llm=llm, background_fn=background_fn)
    epochs = 10
    lr = 1e-4
    for epoch in range(epochs):
        # forward
        loss = 0
        for i in tqdm(range(len(docs))):
            opinions = model(docs[i])
            ypred = [opinion.score for opinion_pair in opinions for opinion in opinion_pair.get_pos_neg_opinions()]
            loss += sum([(y- yp)**2 for y, yp in zip(ys[i], ypred)])
        # backward
        model.zero_grad()            
        loss.backward()
        # update params
        for p in model.parameters():
            p.data += -lr*p.grad
        print(f'epoch {epoch},  iteration: {i}, loss: {loss}')
        pdb.set_trace()
        trace_graph.draw_dot(loss, format='png', output_filepath=f'./data/peru/laws/summaries/epoch_{epoch}')


import argparse
if __name__ == '__main__':
    #model_id = 'meta-llama/Llama-2-70b-chat-hf'
    #model_id = "meta-llama/Llama-2-7b-hf"
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--openai_auth", help="Open AI Key")
    parser.add_argument("-f", "--hf_auth", help="HuggingFace Key")
    args = parser.parse_args()
    run(model_id, args.openai_auth, args.hf_auth)
