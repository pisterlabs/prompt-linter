import cohere
import numpy as np

from cohere.classify import Example

co = cohere.Client('sxxgeiXaO1f9aVYZGBGocCq0pd5R85VcvTNb6ZkG')

def classify(entry):
  response = co.classify(
    model='b73cb43c-740e-4f98-a5c1-90586bbf65e8-ft',
    inputs=[entry])

  predictions = [c.prediction for c in response.classifications]

  # print('The main emotion  of this entry is: ' + predictions[0])
  return predictions[0]
