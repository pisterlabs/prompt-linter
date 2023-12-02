import cohere
from cohere.classify import Example


co = cohere.Client('A48XUypdzQ7MOB2Q4FKIxpIIk5ZmDkchOdaJYlha')
classifications = co.classify(
  model='embed-english-v2.0',
  inputs=["This item was broken when it arrived", "This item broke after 3 weeks"],
  examples=[Example("The order came 5 days early", "positive"), Example("The item exceeded my expectations", "positive"), Example("I ordered more for my friends", "positive"), Example("I would buy this again", "positive"), Example("I would recommend this to others", "positive"), Example("The package was damaged", "negative"), Example("The order is 5 days late", "negative"), Example("The order was incorrect", "negative"), Example("I want to return my item", "negative"), Example("The item\'s material feels low quality", "negative")])
print('The confidence levels of the labels are: {}'.format(
       classifications.classifications))
