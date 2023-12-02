import os
from pathlib import Path

import graphein.molecule as gm
import numpy as np
import pandas as pd
import selfies as sf
import torch
from drfp import DrfpEncoder
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles, rdMolDescriptors
from rxnfp.tokenization import SmilesTokenizer
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator,
    get_default_model_and_tokenizer,
)
from sklearn.feature_extraction.text import CountVectorizer
import openai
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from ast import literal_eval

# Reactions
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    BertModel,
    GPT2Model,
    GPT2Tokenizer,
    AutoModel,
)
from functools import partial


def precalculated(features):
    features = features.apply(literal_eval)
    return features.apply(pd.Series).values


@lru_cache(maxsize=None)
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model=model)
    embedding = response["data"][0]["embedding"]
    return embedding


def ada_embeddings(texts, model="text-embedding-ada-002"):
    """
    Get ADA embeddings for a list of texts.

    :param texts: List of texts to be embedded
    :type texts: list of str
    :param model: Model name to use for embedding (default is "text-embedding-ada-002")
    :type model: str
    :return: NumPy array of ADA embeddings
    """
    get_embedding_with_model = partial(get_embedding, model=model)

    with ProcessPoolExecutor() as executor:
        embeddings = list(
            tqdm(
                executor.map(get_embedding_with_model, texts),
                total=len(texts),
                desc="Getting Embeddings",
            )
        )
    return np.array(embeddings)


# def get_bge_embedding(text, model, tokenizer):
#     # Tokenize sentences
#     encoded_input = tokenizer(
#         text, return_tensors="pt", max_length=512, truncation=True
#     )

#     # Compute token embeddings
#     with torch.no_grad():
#         model_output = model(**encoded_input)
#         # Perform pooling. In this case, cls pooling.
#         sentence_embedding = model_output[0][0, 0]

#     # Normalize embedding
#     sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=0)

#     # Convert PyTorch tensor to NumPy array
#     sentence_embedding_np = sentence_embedding.numpy()

#     return sentence_embedding_np


# def bge_embeddings(texts, model_name="BAAI/bge-large-zh-v1.5"):
#     """
#     Get BGE embeddings for a list of texts.

#     :param texts: List of texts to be embedded
#     :type texts: list of str
#     :param model_name: Pretrained model name to use for embedding
#     :type model_name: str
#     :return: NumPy array of BGE embeddings
#     """

#     # Load model and tokenizer from HuggingFace Hub
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#     model.eval()

#     # Create a partial function with pre-filled arguments for the model and tokenizer
#     partial_get_bge_embedding = partial(
#         get_bge_embedding, model=model, tokenizer=tokenizer
#     )

#     with ProcessPoolExecutor() as executor:
#         embeddings = list(
#             tqdm(
#                 executor.map(partial_get_bge_embedding, texts),
#                 total=len(texts),
#                 desc="Getting BGE Embeddings",
#             )
#         )
#     return np.array(embeddings)


def bge_embeddings(texts, model_name="BAAI/bge-large-zh-v1.5"):
    """
    Get BGE embeddings for a list of texts.

    :param texts: List of texts to be embedded
    :type texts: list of str
    :param model_name: Pretrained model name to use for embedding
    :type model_name: str
    :return: NumPy array of BGE embeddings
    """

    # Load model and tokenizer from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embeddings = []

    for text in tqdm(texts, desc="Getting BGE Embeddings"):
        # Tokenize sentences
        encoded_input = tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embedding = model_output[0][0, 0]

        # Normalize embedding
        sentence_embedding = torch.nn.functional.normalize(
            sentence_embedding, p=2, dim=0
        )

        # Convert PyTorch tensor to NumPy array
        sentence_embedding_np = sentence_embedding.numpy()

        embeddings.append(sentence_embedding_np)

    return np.array(embeddings)


def gte_embeddings(texts, model_name="thenlper/gte-large"):
    """
    Get GTE embeddings for a list of texts.

    :param texts: List of texts to be embedded
    :type texts: list of str
    :param model_name: Pretrained model name to use for embedding
    :type model_name: str
    :return: NumPy array of GTE embeddings
    """

    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # Load model and tokenizer from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Tokenize sentences
    encoded_input = tokenizer(
        texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        outputs = model(**encoded_input)
        # Perform average pooling
        sentence_embeddings = average_pool(
            outputs.last_hidden_state, encoded_input["attention_mask"]
        )

    # Convert PyTorch tensor to NumPy array
    sentence_embeddings_np = sentence_embeddings.numpy()

    return sentence_embeddings_np


def e5_embeddings(texts, model_name="intfloat/e5-large-v2", prefix="query: "):
    """
    Get E5 embeddings for a list of texts.

    :param texts: List of texts to be embedded
    :type texts: list of str
    :param model_name: Pretrained model name to use for embedding
    :type model_name: str
    :param prefix: Prefix to add to each text before embedding. Default is "query: ".
    :type prefix: str
    :return: NumPy array of E5 embeddings
    """

    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # Load model and tokenizer from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Add prefix to each text
    texts = [prefix + text for text in texts]

    # Tokenize sentences
    encoded_input = tokenizer(
        texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        outputs = model(**encoded_input)
        # Perform average pooling
        sentence_embeddings = average_pool(
            outputs.last_hidden_state, encoded_input["attention_mask"]
        )

    # Convert PyTorch tensor to NumPy array
    sentence_embeddings_np = sentence_embeddings.numpy()

    return sentence_embeddings_np


from InstructorEmbedding import INSTRUCTOR


def instructor_embeddings(
    texts,
    model_name="hkunlp/instructor-xl",
    instruction="Represent the chemistry procedure: ",
):
    """
    Get Instructor embeddings for a list of texts.

    :param texts: List of texts to be embedded
    :type texts: list of str
    :param model_name: Pretrained model name to use for embedding
    :type model_name: str
    :param instruction: Instruction string for the embedding task
    :type instruction: str
    :return: NumPy array of Instructor embeddings
    """
    # Load the INSTRUCTOR model
    model = INSTRUCTOR(model_name)

    # Encode the texts with the given instruction
    embeddings = model.encode([[instruction, text] for text in texts])

    # Convert embeddings to NumPy array
    embeddings_np = np.array(embeddings)

    return embeddings_np


def one_hot(df):
    """
    Builds reaction representation as a bit vector which indicates whether
    a certain condition, reagent, reactant etc. is present in the reaction.

    :param df: pandas DataFrame with columns representing different
    parameters of the reaction (e.g. reactants, reagents, conditions).
    :type df: pandas DataFrame
    :return: array of shape [len(reaction_smiles), sum(unique values for different columns in df)]
     with one-hot encoding of reactions
    """
    df_ohe = pd.get_dummies(df)
    return df_ohe.to_numpy(dtype=np.float64)


def rxnfp(reaction_smiles):
    """
    https://rxn4chemistry.github.io/rxnfp/

    Builds reaction representation as a continuous RXNFP fingerprints.
    :param reaction_smiles: list of reaction smiles
    :type reaction_smiles: list
    :return: array of shape [len(reaction_smiles), 256] with rxnfp featurised reactions

    """
    rxn_model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(rxn_model, tokenizer)
    rxnfps = [rxnfp_generator.convert(smile) for smile in reaction_smiles]
    return np.array(rxnfps, dtype=np.float64)


def rxnfp2(reaction_smiles):
    print(os.getcwd())
    model_path = "../rxn_yields/trained_models/uspto/uspto_milligram_smooth_random_test_epochs_2_pretrained/checkpoint-30204-epoch-2/"
    tokenizer_vocab_path = "../rxn_yields/trained_models/uspto/uspto_milligram_smooth_random_test_epochs_2_pretrained/checkpoint-30204-epoch-2/vocab.txt"

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    model = BertModel.from_pretrained(model_path)
    model = model.eval()
    model.to(device)

    tokenizer = SmilesTokenizer(tokenizer_vocab_path)

    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    rxnfps = [rxnfp_generator.convert(smile) for smile in reaction_smiles]
    return np.array(rxnfps, dtype=np.float64)


def drfp(reaction_smiles, bond_radius=3, nBits=2048):
    """
    https://github.com/reymond-group/drfp

    Builds reaction representation as a binary DRFP fingerprints.
    :param reaction_smiles: list of reaction smiles
    :type reaction_smiles: list
    :return: array of shape [len(reaction_smiles), nBits] with drfp featurised reactions

    """
    fps = DrfpEncoder.encode(reaction_smiles, n_folded_length=nBits, radius=bond_radius)
    print(np.array(fps, dtype=np.float64).shape, "drfp vectors shape")

    return np.array(fps, dtype=np.float64)


def gpt2(reaction_smiles):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)

    def get_sentence_vector(reaction, tokenizer, model):
        tokens = tokenizer(reaction, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
        hidden_states = outputs.hidden_states[-1]
        sentence_vector = hidden_states.mean(dim=1).squeeze().numpy().astype(np.float64)
        return sentence_vector

    reaction_vectors = [
        get_sentence_vector(reaction, tokenizer, model) for reaction in reaction_smiles
    ]
    print(np.array(reaction_vectors).shape, "reaction vectors shape")
    return np.array(reaction_vectors)


def drxnfp(reaction_smiles, bond_radius=3, nBits=2048):
    drfps = drfp(reaction_smiles, nBits, radius=bond_radius)
    rxnfps = rxnfp(reaction_smiles)
    return np.concatenate([drfps, rxnfps], axis=1)


def drfpfingerprints(reaction_smiles, smiles, bond_radius=3, nBits=2048):
    drfps = drfp(reaction_smiles, nBits, radius=bond_radius)
    fingerprints = fingerprints(smiles, bond_radius=bond_radius, nBits=nBits)
    return np.concatenate([drfps, fingerprints], axis=1)


# Molecules
def fingerprints(smiles, bond_radius=3, nBits=2048):
    """
    Get Morgan fingerprints for a list of SMILES strings.
    """
    rdkit_mols = [MolFromSmiles(smile) for smile in smiles]
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=nBits)
        for mol in rdkit_mols
    ]
    return np.asarray(fingerprints)


# auxiliary function to calculate the fragment representation of a molecule
def fragments(smiles):
    # descList[115:] contains fragment-based features only
    # (https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html)
    # Update: in the new RDKit version the indices are [124:]
    fragments = {d[0]: d[1] for d in Descriptors.descList[124:]}
    frags = np.zeros((len(smiles), len(fragments)))
    for i in range(len(smiles)):
        mol = MolFromSmiles(smiles[i])
        try:
            features = [fragments[d](mol) for d in fragments]
        except:
            raise Exception("molecule {}".format(i) + " is not canonicalised")
        frags[i, :] = features

    return frags


# auxiliary function to calculate bag of character representation of a molecular string
def bag_of_characters(smiles, max_ngram=5, selfies=False):
    if selfies:  # convert SMILES to SELFIES
        strings = [sf.encoder(smiles[i]) for i in range(len(smiles))]
    else:  # otherwise stick with SMILES
        strings = smiles

    # extract bag of character (boc) representation from strings
    cv = CountVectorizer(ngram_range=(1, max_ngram), analyzer="char", lowercase=False)
    return cv.fit_transform(strings).toarray()


def mqn_features(smiles):
    """
    Builds molecular representation as a vector of Molecular Quantum Numbers.
    :param reaction_smiles: list of molecular smiles
    :type reaction_smiles: list
    :return: array of mqn featurised molecules
    """
    molecules = [MolFromSmiles(smile) for smile in smiles]
    mqn_descriptors = [rdMolDescriptors.MQNs_(molecule) for molecule in molecules]
    return np.asarray(mqn_descriptors)


def chemberta_features(smiles):
    # any model weights from the link above will work here
    model = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    tokenized_smiles = [tokenizer(smile, return_tensors="pt") for smile in smiles]
    outputs = [
        model(
            input_ids=tokenized_smile["input_ids"],
            attention_mask=tokenized_smile["attention_mask"],
            output_hidden_states=True,
        )
        for tokenized_smile in tokenized_smiles
    ]
    embeddings = torch.cat(
        [output["hidden_states"][0].sum(axis=1) for output in outputs], axis=0
    )
    return embeddings.detach().numpy()


def graphs(smiles, graphein_config=None):
    return [gm.construct_graph(smiles=i, config=graphein_config) for i in smiles]


def cddd(smiles):
    current_path = os.getcwd()
    os.chdir(Path(os.path.abspath(__file__)).parent)
    cddd = pd.read_csv("precalculated_featurisation/cddd_additives_descriptors.csv")
    cddd_array = np.zeros((cddd.shape[0], 512))
    for i, smile in enumerate(smiles):
        row = cddd[cddd["smiles"] == smile][cddd.columns[3:]].values
        cddd_array[i] = row
    os.chdir(current_path)
    return cddd_array


def xtb(smiles):
    current_path = os.getcwd()
    os.chdir(Path(os.path.abspath(__file__)).parent)
    xtb = pd.read_csv("precalculated_featurisation/xtb_qm_descriptors_2.csv")
    xtb_array = np.zeros((xtb.shape[0], len(xtb.columns[:-2])))
    for i, smile in enumerate(smiles):
        row = xtb[xtb["Additive_Smiles"] == smile][xtb.columns[:-2]].values
        xtb_array[i] = row
    os.chdir(current_path)
    return xtb_array


def random_features():
    # todo random continous random bit vector
    pass
