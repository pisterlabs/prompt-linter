from ..client import Completion_Client
from ....mixins import Example

# Simple classifier app :)
if __name__ == "__main__":
    e1 = Example('Tweet: "I loved the new Batman movie!"', 'Sentiment: "Positive"')
    e2 = Example('Tweet: "I hated that ice cream."', 'Sentiment: "Negative"')

    # API Key is read from OPENAI_API_KEY
    client = Completion_Client("Decide whether a Tweet's sentiment is positive, neutral, or negative.", [e1,e2])
    
    # Add another example
    client.add_examples(Example('Tweet: "I did not like the explosion"', 'Sentiment: "Negative"'))

    # Get tweet to classify
    tweet_to_classify = "Amusement parks are ok"

    # Print the tweet to classify
    print(f"\nClassifying tweet: '{tweet_to_classify}'")

    # Send to the model with examples
    response = client.run_prompt(tweet_to_classify)

    # Print result
    print(response)
