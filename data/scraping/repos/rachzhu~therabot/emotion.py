import cohere
from cohere.classify import Example
co = cohere.Client('cmeanyUvNYBwCIqIVbsovYoM8hkgCQllXYq7AaVF')
response = co.classify(
  model='large',
  inputs=["I feel anxious"],
  examples=[Example("I'm feeling suicidal", "Suicide"),
            Example("The world would be a better place without me", "Suicide"),
            Example("Life is meaningless", "Depression"),
            Example("I can't get out of bed", "Depression"),
            Example("I feel like something bad is about to happen", "Anxiety"),
            Example("My heart is beating fast and it's hard to breathe", "Anxiety"),
            Example("I can't stop drinking", "Addiction"),
            Example("I'm becoming reliant on substances", "Addiction"),
            Example("I'm having hallucinations", "Schizophrenia"),
            Example("It's hard for my to understand things", "Schizophrenia")])
print('The confidence levels of the labels are: {}'.format(response.classifications))
