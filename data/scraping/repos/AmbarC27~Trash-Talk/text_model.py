import cohere
from cohere.classify import Example
import examples
co = cohere.Client('DLHq9Go9DHorXORElej8QEbINQZYq3NrM0G3RsGo')

examples = examples.black_examples + examples.blue_examples + examples.green_examples

def getBinColourFromText(item):
    response = co.classify(
        model='large',
        inputs=[item],
        examples=examples)
    return(response.classifications[0].prediction)
# print("Formatted version: {}".format(response.classifications)[0])
# print("Formatted version: {}".response.classifications[0])

#print(response.classifications[0].prediction)
#print(response.classifications[0].confidence)
#print(response.classifications[0].labels)
