from typing import List, Tuple
from langchain import LLMBashChain, LLMChain, OpenAI, PromptTemplate
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import custom_types
from sklearn.cluster import KMeans


# A list of tuples, where the first element is the label and the second is the embedding

def cluster(embeddings: custom_types.Embeddings, n_clusters: int, dimensions=None):
    """
    Cluster the embeddings using k-means clustering.
    :param embeddings: A list of tuples, where the first element is the label and the second is the embedding
    """
    data = np.array([e[1] for e in embeddings])
    if dimensions is None:
        dimensions = len(data[0])
        df = np.array([np.array(d) for d in data])
    else:
        pca = PCA(dimensions)
        # Transform the data
        df = pca.fit_transform(data)
    
    
    #Initialize the class object
    kmeans = KMeans(n_clusters=n_clusters)
    
    #predict the labels of clusters.
    label = kmeans.fit_predict(df)
    
    #Getting unique labels
    u_labels = np.unique(label)
    return df, label, u_labels


# def relative_labels(thms1: List[str], thms2: List[str], centroid_idx1: int, centroid_idx2: int, llm) -> str:
#     template = """Given the following two labeled sets of Lean theorems, can you describe the main difference in one sentence?

# Set {label1}: "{set1}"

# Set {label2}: "{set2}"
# """
#     prompt = PromptTemplate(template=template, input_variables=["set1", "set2", "label1", "label2"])
#     llm_chain = LLMChain(prompt=prompt, llm=llm)
#     return llm_chain.run(set1="\n".join(thms1), set2="\n".join(thms2), label1=f"Cluster {centroid_idx1}", label2=f"Cluster {centroid_idx2}")
    

async def local_neighbor_with_descr_labels(thms_node: List[str], descr_node: str, thms_local: List[List[str]], descr_thms_local: List[str], llm: LLMBashChain):
    merged_non_prim = [f"Description: {descr_thms_local[i]}\n" + "\n".join(t) for i, t in enumerate(thms_local)] if descr_thms_local[0] != "" \
        else ["\n".join(t) for t in thms_local]
    joined_non_prim = "\n\n".join(merged_non_prim)

    joined_prim = (f"Description: {descr_node}" + "\n" if descr_node != "" else "") + "\n".join(thms_node)
    prompt = f"""You will be given a set of non-primary theorems and a set of primary theorems{ " as well as descriptions for both" if descr_node[0] != "" else ""}. Can you briefly discuss the main focus of the primary theorems and how it differs from the remaining theorems?
Assume that when the descriptions are given for the non-primary theorems, that they do not reference the set of primary theorems at all.

Non-primary theorems: "{joined_non_prim}"

Primary theorems: "{joined_prim}"

RESPONSE:
"""
    r = await llm.agenerate([prompt])
    return r.generations[0][0].text
    
