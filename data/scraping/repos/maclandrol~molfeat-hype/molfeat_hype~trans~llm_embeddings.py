from typing import Union
from typing import List
from typing import Optional

import os
import copy
import requests
import dotenv
import datamol as dm
import contextlib

from loguru import logger
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings as LangChainEmbeddings
from molfeat.trans.pretrained.base import PretrainedMolTransformer
from molfeat_hype.utils import convert_smiles
from molfeat_hype.utils import create_symlink
from molfeat_hype.utils import CACHE_DIR

dotenv.load_dotenv()


class LLMTransformer(PretrainedMolTransformer):
    """
    Large Language Model Embeddings Transformer for molecule.
    This transformer embeds molecules using available Large Language Models (LLMs) through langchain.
    Please note that the LLMs do not have any molecular context as they were not trained on molecules or any specific molecular task.
    They are just trained on a large corpus of text.


    !!! warning "Caching computation"
        LLMs can be computationally expensive and even financially expensive if you use the OpenAI embeddings.
        To avoid recomputing the embeddings for the same molecules, we recommend using a molfeat Cache object.
        By default, an in-memory cache (DataCache) is used, but other caching systems can be explored.


    ??? note "Using OpenAI Embeddings"
        If you are using the OpenAI embeddings, you need to provide an 'open_ai_key' argument or define one through an environment variable 'OPEN_AI_KEY'.
        Please note that only the `text-embedding-ada-002` model is supported.
        Refer to OpenAI's documentation for more information.

    ??? note "Using LLAMA Embeddings"
        The Llama embeddings are provided via the python bindings of `llama.cpp`.
        We do not provide the path to the quantized Llama model. However, it's easy to find them online;
        some people have shared the torrent/IPFS/direct download links to the Llama weights, then you can quantized them yourself.

    ??? note "Using Sentence Transformer Embeddings"
        The sentence transformer embeddings are based on the SentenceTransformers package.

    """

    SUPPORTED_EMBEDDINGS = [
        "openai/text-embedding-ada-002",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "llama.cpp",
    ]

    def __init__(
        self,
        kind=Union[str, LangChainEmbeddings],
        standardize: bool = True,
        precompute_cache: bool = True,
        n_jobs: int = 0,
        dtype=float,
        openai_api_key: Optional[str] = None,
        quantized_model_path: Optional[str] = None,
        parallel_kwargs: Optional[dict] = None,
        **params,
    ):
        """Instantiate a LLM Embeddings transformer

        Args:
            kind: kind of LLM to use. Supported LLMs are accessible through the SUPPORTED_EMBEDDINGS attribute. Here are a few:
                - "openai/text-embedding-ada-002"
                - "sentence-transformers/all-MiniLM-L6-v2"
                - "sentence-transformers/all-mpnet-base-v2"
                - "llama.cpp"
                You can also provide any model hosted on hugginface that compute embeddings
            standardize: if True, standardize smiles before embedding
            precompute_cache: if True, add a cache to cache the embeddings for the same molecules.
            n_jobs: number of jobs to use for preprocessing smiles.
            dtype: data type to use for the embeddings return type
            openai_api_key: openai api key to use. If None, will try to get it from the environment variable OPENAI_API_KEY
            quantized_model_path: path to the quantized model for llama.cpp. If None, will try to get it from the environment variable quantized_MODEL_PATH
            **params: parameters to pass to the LLM embeddings. See langchain documentation
        """

        self.kind = kind
        self.model = None
        self.standardize = standardize
        if isinstance(kind, str):
            if not kind.startswith(tuple(self.SUPPORTED_EMBEDDINGS)):
                logger.warning(f"Model {kind} not found, trying from huggingface hub.")
                on_hgf = requests.get(f"https://huggingface.co/{kind}")
                try:
                    on_hgf.raise_for_status()
                except:
                    raise ValueError(
                        f"Unknown LLM type {kind} requested. Supported models are {self.SUPPORTED_EMBEDDINGS}"
                    )
            if kind.startswith("openai/"):
                if openai_api_key is None:
                    openai_api_key = os.environ.get("OPENAI_API_KEY")
                self.model = OpenAIEmbeddings(
                    model=kind.replace("openai/", ""), openai_api_key=openai_api_key, **params
                )
            elif kind.startswith("llama.cpp"):
                if quantized_model_path is None:
                    quantized_model_path = os.environ.get("QUANT_MODEL_PATH")
                if quantized_model_path is not None and dm.fs.exists(quantized_model_path):
                    create_symlink(quantized_model_path, kind)
                else:
                    model_base_name = os.path.splitext(os.path.basename(quantized_model_path))[0]
                    quantized_model_path = dm.fs.glob(dm.fs.join(CACHE_DIR, f"{model_base_name}*"))
                    if len(quantized_model_path) == 0:
                        raise ValueError(
                            f"Could not find the quantized model {model_base_name} anywhere, including in the cache dir {CACHE_DIR}"
                        )
                    quantized_model_path = quantized_model_path[0]
                with contextlib.redirect_stdout(None):
                    with contextlib.redirect_stderr(None):
                        n_ctx = max(params.get("n_ctx", 1024), 1024)
                        params["n_ctx"] = n_ctx
                        self.model = LlamaCppEmbeddings(model_path=quantized_model_path, **params)
                        self.model.client.verbose = False
            else:
                self.model = HuggingFaceEmbeddings(
                    model_name=kind, model_kwargs=params, cache_folder=CACHE_DIR
                )
        super().__init__(
            precompute_cache=precompute_cache,
            n_jobs=n_jobs,
            dtype=dtype,
            device="cpu",
            parallel_kwargs=parallel_kwargs,
            **params,
        )

    def _convert(self, inputs: List[Union[str, dm.Mol]], **kwargs):
        """Convert the list of input molecules into the proper format for embeddings

        Args:
            inputs: list of input molecules
            **kwargs: additional keyword arguments for API consistency

        """
        self._preload()
        parallel_kwargs = copy.deepcopy(getattr(self, "parallel_kwargs", {}))
        parallel_kwargs["n_jobs"] = self.n_jobs
        return convert_smiles(inputs, parallel_kwargs, standardize=self.standardize)

    def _embed(self, smiles: List[str], **kwargs):
        """This function takes a list of smiles or molecules and return the featurization
        corresponding to the inputs.
        In `transform` and `_transform`, this function is called after calling `_convert`

        Args:
            smiles: input smiles
            **kwargs: additional keyword arguments for API consistency
        """
        return self.model.embed_documents(smiles)
