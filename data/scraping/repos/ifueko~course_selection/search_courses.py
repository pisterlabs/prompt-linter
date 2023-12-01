import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity

def toarray(x):
   x = [float(v.strip()) for v in x.strip('[').strip(']').split(',')]
   return x

def search_courses(df, query, n=10):
   embedding = query.embedding
   df['similarities'] = df.embedding_combined.apply(lambda x: cosine_similarity(np.asarray(toarray(x), dtype='float64'), np.asarray(embedding, dtype='float64')))
   res = df.sort_values('similarities', ascending=False).head(n)
   print(res)
   return res
