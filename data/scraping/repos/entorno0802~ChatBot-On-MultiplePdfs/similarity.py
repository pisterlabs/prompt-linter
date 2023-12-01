import numpy as np
import cohere as co

sentence1 = np.array(co.embed(["I like to be in my house"]).embeddings)
sentence2 = np.array(co.embed(["I enjoy staying home"]).embeddings)
sentence3 = np.array(co.embed(["the isotope 238u decays to 206pb"]).embeddings)

print(sentence1)
print(sentence2)
print(sentence3)
