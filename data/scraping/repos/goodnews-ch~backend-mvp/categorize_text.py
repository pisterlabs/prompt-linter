import cohere 
from cohere.classify import Example 
co = cohere.Client('3Gadb4V5oKd2YIwc6rz7Oxw6LOYFTxFSbg0nxy7k')

def categorize_text(text_input):
    response = co.classify(
        model='841e59dd-dd43-4bb6-8747-cbd5ff8f1499-ft',
        inputs=[text_input]
    )
    print("The text's classification was '{}'".format(response.classifications[0].prediction))
    return response.classifications[0].prediction
