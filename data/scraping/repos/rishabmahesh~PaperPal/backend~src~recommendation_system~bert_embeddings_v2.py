from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import openai

def pad_word_embeddings(word_embeddings, max_length):
    """
    Pad word embeddings with zeros to ensure a fixed sequence length.

    Args:
        word_embeddings (torch.Tensor): Tensor of word embeddings with shape (batch_size, sequence_length, embedding_dim).
        max_length (int): Maximum sequence length to pad to.

    Returns:
        torch.Tensor: Padded word embeddings with shape (batch_size, max_length, embedding_dim).
    """
    if word_embeddings.shape[1] < max_length:
        padding = torch.zeros((word_embeddings.shape[0], max_length - word_embeddings.shape[1], word_embeddings.shape[2]))
        word_embeddings = torch.cat((word_embeddings, padding), dim=1)
    return word_embeddings


# from src.data.query_helper import get_all_abstracts
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."

abstract1 = """Large labeled training sets are the critical building blocks of supervised learning methods and are key enablers of deep learning techniques. For some applications, creating labeled training sets is the most time-consuming and expensive part of applying machine learning. We therefore propose a paradigm for the programmatic creation of training sets called data programming in which users express weak supervision strategies or domain heuristics as labeling functions, which are programs that label subsets of the data, but that are noisy and may conflict. We show that by explicitly representing this training set labeling process as a generative model, we can "denoise" the generated training set, and establish theoretically that we can recover the parameters of these generative models in a handful of settings. We then show how to modify a discriminative loss function to make it noise-aware, and demonstrate our method over a range of discriminative models including logistic regression and LSTMs. Experimentally, on the 2014 TAC-KBP Slot Filling challenge, we show that data programming would have led to a new winning score, and also show that applying data programming to an LSTM model leads to a TAC-KBP score almost 6 F1 points over a state-of-the-art LSTM baseline (and into second place in the competition). Additionally, in initial user studies we observed that data programming may be an easier way for non-experts to create machine learning models when training data is limited or unavailable."""
# abstract2 = """Labeling training data is increasingly the largest bottleneck in deploying machine learning systems. We present Snorkel, a first-of-its-kind system that enables users to train state-of-the-art models without hand labeling any training data. Instead, users write labeling functions that express arbitrary heuristics, which can have unknown accuracies and correlations. Snorkel denoises their outputs without access to ground truth by incorporating the first end-to-end implementation of our recently proposed machine learning paradigm, data programming. We present a flexible interface layer for writing labeling functions based on our experience over the past year collaborating with companies, agencies, and research labs. In a user study, subject matter experts build models 2.8x faster and increase predictive performance an average 45.5% versus seven hours of hand labeling. We study the modeling tradeoffs in this new setting and propose an optimizer for automating tradeoff decisions that gives up to 1.8x speedup per pipeline execution. In two collaborations, with the U.S. Department of Veterans Affairs and the U.S. Food and Drug Administration, and on four open-source text and image data sets representative of other deployments, Snorkel provides 132% average improvements to predictive performance over prior heuristic approaches and comes within an average 3.60% of the predictive performance of large hand-curated training sets."""
abstract2 = """UD Co-Spaces (Urban Design Collaborative Spaces) is an in- tegrated, tabletop-centered multi-display environment for en- gaging the public in the complex process of collaborative ur- ban design. We describe the iterative user-centered process that we followed over six years through a close interdisci- plinary collaboration involving experts in urban design and neighbourhood planning. Versions of UD Co-Spaces were deployed in five real-world charrettes (planning workshops) with 83 participants, a heuristic evaluation with three domain experts, and a qualitative laboratory study with 37 partici- pants. We reflect on our design decisions and how multi- display environments can engage a broad range of stake- holders in decision making and foster collaboration and co- creation within urban design. We examine the parallel use of different displays, each with tailored interactive visualiza- tions, and whether this affects what people can learn about the consequences of their choices for sustainable neighborhoods. We assess UD Co-Spaces using seven principles for collabo- rative urban design tools that we identified based on literature in urban design, CSCW, and public engagement."""
# abstract3 = """Intelligence — the ability to learn, reason and solve problems — is at the forefront of behavioural genetic research. Intelligence is highly heritable and predicts important educational, occupational and health outcomes better than any other trait. Recent genome-wide association studies have successfully identified inherited genome sequence differences that account for 20% of the 50% heritability of intelligence. These findings open new avenues for research into the causes and consequences of intelligence using genome-wide polygenic scores that aggregate the effects of thousands of genetic variants. In this Review, we highlight the latest innovations and insights from the genetics of intelligence and their applications and implications for science and society."""
abstract3 = """Past studies have shown that when a visualization uses pictographs to encode data, they have a positive effect on memory, engagement, and assessment of risk. However, little is known about how pictographs affect one’s ability to understand a visualization, beyond memory for values and trends. We conducted two crowdsourced experiments to compare the effectiveness of using pictographs when showing part-to-whole relationships. In Experiment 1, we compared pictograph arrays to more traditional bar and pie charts. We tested participants’ ability to generate high-level insights following Bloom’s taxonomy of educational objectives via 6 free-response questions. We found that accuracy for extracting information and generating insights did not differ overall between the two versions. To explore the motivating differences between the designs, we conducted a second experiment where participants compared charts containing pictograph arrays to more traditional charts on 5 metrics and explained their reasoning. We found that some participants preferred the way that pictographs allowed them to envision the topic more easily, while others preferred traditional bar and pie charts because they seem less cluttered and faster to read. These results suggest that, at least in simple visualizations depicting part-to-whole relationships, the choice of using pictographs has little influence on sensemaking and insight extraction. When deciding whether to use pictograph arrays, designers should consider visual appeal, perceived comprehension time, ease of envisioning the topic, and clutteredness."""

def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


sentence_embedding1 = get_embedding(abstract1)
sentence_embedding2 = get_embedding(abstract2)
sentence_embedding3 = get_embedding(abstract3)

#
# # Tokenize input sentences
# encoded_input1 = tokenizer(abstract1, return_tensors='pt', padding=True, truncation=True)
# encoded_input2 = tokenizer(abstract2, return_tensors='pt', padding=True, truncation=True)
# encoded_input3 = tokenizer(abstract3, return_tensors='pt', padding=True, truncation=True)
#
# # Extract word embeddings
# word_embeddings1 = model(**encoded_input1).last_hidden_state
# word_embeddings2 = model(**encoded_input2).last_hidden_state
# word_embeddings3 = model(**encoded_input3).last_hidden_state
#
# # Perform min-max pooling aggregation
# # sentence_embedding1 = np.concatenate((word_embeddings1.min(axis=1), word_embeddings1.max(axis=1)), axis=1)
# # sentence_embedding2 = np.concatenate((word_embeddings2.min(axis=1), word_embeddings2.max(axis=1)), axis=1)
# # sentence_embedding3 = np.concatenate((word_embeddings3.min(axis=1), word_embeddings3.max(axis=1)), axis=1)
#
# # Perform averaging aggregation
# # sentence_embedding1 = word_embeddings1.mean(axis=1)
# # sentence_embedding2 = word_embeddings2.mean(axis=1)
# # sentence_embedding3 = word_embeddings3.mean(axis=1)
#
# # Manual padding to ensure same sequence length
# max_length = max(word_embeddings1.shape[1], word_embeddings2.shape[1], word_embeddings3.shape[1])
# word_embeddings1 = pad_word_embeddings(word_embeddings1, max_length)
# word_embeddings2 = pad_word_embeddings(word_embeddings2, max_length)
# word_embeddings3 = pad_word_embeddings(word_embeddings3, max_length)
#
# # Perform self-attention mechanism
# attention_weights1 = torch.nn.functional.softmax(torch.matmul(word_embeddings1, word_embeddings1.transpose(1, 2)), dim=2)
# attention_weights2 = torch.nn.functional.softmax(torch.matmul(word_embeddings2, word_embeddings2.transpose(1, 2)), dim=2)
# attention_weights3 = torch.nn.functional.softmax(torch.matmul(word_embeddings3, word_embeddings3.transpose(1, 2)), dim=2)
#
# # Apply self-attention weights to word embeddings
# self_attention_embeddings1 = torch.matmul(attention_weights1, word_embeddings1)
# self_attention_embeddings2 = torch.matmul(attention_weights2, word_embeddings2)
# self_attention_embeddings3 = torch.matmul(attention_weights3, word_embeddings3)
#
# # Aggregate embeddings using mean pooling
# sentence_embedding1 = torch.mean(self_attention_embeddings1, dim=1).detach().numpy()
# sentence_embedding2 = torch.mean(self_attention_embeddings2, dim=1).detach().numpy()
# sentence_embedding3 = torch.mean(self_attention_embeddings3, dim=1).detach().numpy()
#


# Compute cosine similarity
cs_12 = cosine_similarity([sentence_embedding1], [sentence_embedding2])
cs_13 = cosine_similarity([sentence_embedding1], [sentence_embedding3])
cs_23 = cosine_similarity([sentence_embedding2], [sentence_embedding3])

print("Cosine similarity between sentence 1 and sentence 2:", cs_12)
print("Cosine similarity between sentence 1 and sentence 3:", cs_13)
print("Cosine similarity between sentence 2 and sentence 3:", cs_23)
