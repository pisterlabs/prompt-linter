import openai
from annoy import AnnoyIndex
from pprint import pprint

# sample corpus
texts = [
"We report the development of GPT-4, a large-scale, multimodal model which can accept"
  "image and text inputs and produce text outputs. While less capable than humans in"
  "many real-world scenarios, GPT-4 exhibits human-level performance on various professional"
  "and academic benchmarks, including passing a simulated bar exam with a score around"
  "the top 10% of test takers.",
  "While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level"
  "performance on various professional and academic benchmarks, including passing a"
  "simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-"
  "based model pre-trained to predict the next token in a document."
]

# from texts to embeddings (numeric representations of the texts)
embeddings = [
    r['embedding']
    for r in openai.Embedding.create(input = texts, model='text-embedding-ada-002')['data']
] 
hidden_size = len(embeddings[0])
index = AnnoyIndex(hidden_size, 'angular')  #  "angular" =  cosine
for i, e in enumerate(embeddings): 
    index.add_item(i , e)
index.build(10)  # build 10 trees for efficient search

query = "Is GPT4 as smart as humans?"
embedding =  openai.Embedding.create(input = [query], model='text-embedding-ada-002')['data'][0]['embedding']

# get nearest neighbors by vectors
indices, distances = index.get_nns_by_vector(embedding, n=10, include_distances=True)


results =  [ 
    (texts[i], d)
    for i, d in zip(indices, distances)
]

pprint(results)

"""
[('While less capable than humans in many real-world scenarios, GPT-4 exhibits '
  'human-levelperformance on various professional and academic benchmarks, '
  'including passing asimulated bar exam with a score around the top 10% of '
  'test takers. GPT-4 is a Transformer-based model pre-trained to predict the '
  'next token in a document.',
  0.4776711165904999),
 ('We report the development of GPT-4, a large-scale, multimodal model which '
  'can acceptimage and text inputs and produce text outputs. While less '
  'capable than humans inmany real-world scenarios, GPT-4 exhibits human-level '
  'performance on various professionaland academic benchmarks, including '
  'passing a simulated bar exam with a score aroundthe top 10% of test takers.',
  0.4969678521156311)]
"""

