import sys
import cohere
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

co = cohere.Client('L2VMOXwleskZQjVuP5QEe2puJTKNLAzGaRhSEVTK')

print(sys.argv[1])
print(sys.argv[2])

# embedding for user answer: sentence1
sentence1 = np.array(co.embed([sys.argv[1]]).embeddings)
# embedding for tutor answer: sentence2
sentence2 = np.array(co.embed([sys.argv[2]]).embeddings)

# cosine similarity between sentence1 and sentence2
rating = cosine_similarity(sentence1, sentence2)[0][0]
if rating >= 1.3 or rating <= 0.7:
  print("Incorrect")
else:
  print("Correct")