
"""Language Model Tools"""

import asyncio, os

from typing import Callable, Optional, Literal
from .lmInterface import StreamData, Iterator
from .lmInterface import getStreamIter, streamOutput, ChatStreamIterType
from .utils import autoTorchDevice, MuteEverything, Timer

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def summarize(txt: str) -> Iterator[StreamData]:
    from .lmInterface import OpenAIChatStreamIter
    prompt = f"Summarize the following paper, "\
        "your summary should focus on the motivation, methods and contributions of the paper. "\
        "Your summary should be in academic literature style. Don't exagerate the significance. "\
        f"Here is the paper: {txt}"
    return OpenAIChatStreamIter()(prompt)

dummy_print = lambda x, end = " ", flush = True: ...
async def structuredSummerize(txt: str, model: ChatStreamIterType = "gpt-3.5-turbo", print_func: Callable = dummy_print) -> str:
    PRINT = True
    if print_func is dummy_print:
        PRINT = False

    ai = getStreamIter(model)
    ai.conversations.system = "A conversation between a human and an AI research assistant. "\
        "The human is asking the AI to summarize a paper."\
        "The AI gives short and conscise response in academic literature style. "\

    prompt = "Please summarize the following paper in about 200 words, "\
        "your summary should be 3 paragraphs, each paragraph should start with \'Background and Motivations: \', "\
        "\'Methods: \' and \'Contributions: \' respectively. "\
        "No need to mention the title in your summary."\
        f"Here is the paper: {txt}"
    return streamOutput(ai(prompt), print_func)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
g_auto_tokenizers = {}
g_auto_models = {}
EncoderT = Literal["distilbert-base-uncased", "yikuan8/Clinical-Longformer", "allenai/longformer-base-4096", "bert-base-uncased", "sentence-transformers/all-mpnet-base-v2"]
_default_encoder = "sentence-transformers/all-mpnet-base-v2"
# reference: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
@torch.inference_mode()
async def vectorize(
    txt: str, 
    model_name: EncoderT = _default_encoder,
    max_len: Optional[int] = None, 
    verbose = False
    ) -> torch.Tensor:
    """
    max_len: the max length of the input text, the rest will be truncated, automatically set to the length of the model if None
    take a long text and return a vector of shape [1, d_feature]
    """
    global g_auto_tokenizers, g_auto_models
    device = autoTorchDevice()
    txt = txt.replace("\n", " ")
    txt = " ".join(txt.split())

    def getTokenizerModel(model_name, device):
        with Timer("Loading text encoder model"):
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _model = AutoModel.from_pretrained(model_name).to(device)
        return _tokenizer, _model

    try:
        g_auto_models[model_name]
    except KeyError:
        _tokenizer, _model = getTokenizerModel(model_name, device)
        g_auto_tokenizers[model_name] = _tokenizer
        g_auto_models[model_name] = _model
    auto_tokenizer = g_auto_tokenizers[model_name]
    auto_model = g_auto_models[model_name]

    if max_len is None:
        # if model_name == "distilbert-base-uncased": max_len = 512
        # elif model_name == "bert-base-uncased": max_len = 512
        # elif model_name == "yikuan8/Clinical-Longformer": max_len = 4096
        # elif model_name == "allenai/longformer-base-4096": max_len = 4096
        # elif model_name == "sentence-transformers/all-mpnet-base-v2": max_len = 512
        max_len = auto_tokenizer.model_max_length
    

    # Tokenize sentences
    encoded_input = auto_tokenizer(txt, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = auto_model(**encoded_input)

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # import pdb; pdb.set_trace()
    with MuteEverything(enable=not verbose):
        if len(encoded_input['input_ids'][0]) == max_len:   # type: ignore
            print("Warning<vectorize>: input text is too long, truncated.")
    return sentence_embeddings

async def featurize(
        txt: str, 
        word_chunk: int = 256, 
        model_name: EncoderT = _default_encoder, 
        dim_reduce: bool = False,
        verbose = False) -> torch.Tensor:
    """
    take a long text return it's feature
    if dim_reduce is True, return a tensor of shape [d_feature]
    else, return a tensor of shape [n, d_feature]
    """
    assert txt != "", "txt cannot be empty"
    txt_split = txt.split()
    txt_chunks = [" ".join(txt_split[i:i+word_chunk]) for i in range(0, len(txt_split), word_chunk)]
    _vec_chunk_tasks = [vectorize(chunk, verbose=verbose, model_name=model_name) for chunk in txt_chunks]
    vec_chunks: list[torch.Tensor] = await asyncio.gather(*_vec_chunk_tasks)
    feats = torch.cat(vec_chunks, dim=0)    # [n, d_feature]

    if not dim_reduce:
        return feats
    else:
        if len(feats) == 1:
            return feats[0]

    last_chunk_len = len(txt_chunks[-1].split())
    last_chunk_weight = last_chunk_len / word_chunk

    # weighted average
    feat = torch.sum(feats[:-1], dim=0) + feats[-1] * last_chunk_weight
    return feat/(len(feats)-1+last_chunk_weight)
