import cohere
import pandas as pd
from cohere.responses.classify import Example

# Initialize the Cohere client with your API key
co = cohere.Client('YOUR_API_KEY')

# Define your input title
input_title = str(input("Enter your news title: "))

# Load fake and real news data from CSV files
fake = pd.read_csv('Fake.csv')  # Replace 'FakeTitles.csv' with your fake news titles file
real = pd.read_csv('True.csv')  # Replace 'RealTitles.csv' with your real news titles file

# Create instances of the Example class for fake and real news titles
fake_examples = [Example(title, label="fake") for title in fake['title']]
real_examples = [Example(title, label="real") for title in real['title']]

# Combine both fake and real examples
examples = fake_examples + real_examples

# Call the classify method with the examples list
response = co.classify(examples=examples, inputs=[input_title])

# Print the response
print(response.classifications[0].prediction)
print(response.classifications[0].confidence)
