from cohere.responses.classify import Example

SENTIMENT_EXAMPLES = [
    Example("I love this product", "positive"),
    Example("The UI is great", "positive"),
    Example("I ordered more for my friends", "positive"),
    Example("I would buy this again", "positive"),
    Example("I would recommend this to others", "positive"),
    Example("I don't like the product", "negative"),
    Example("I'm struggling", "negative"),
    Example("The order was incorrect", "negative"),
    Example("I want to return my item", "negative"),
    Example("The item's material feels low quality", "negative"),
    Example("The product was okay", "neutral"),
    Example("I received five items in total", "neutral"),
    Example("I bought it from the website", "neutral"),
    Example("I used the product this morning", "neutral"),
    Example("The product arrived yesterday", "neutral"),
]
