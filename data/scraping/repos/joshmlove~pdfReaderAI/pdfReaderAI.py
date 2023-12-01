import openai
import pdfplumber
import constants

# Load the OpenAI API key
openai.api_key = constants.APIKEY

# Read your own data from the PDF file
with pdfplumber.open('Josh_Love_Resume copy.pdf') as pdf:
    data = ' '.join(page.extract_text() for page in pdf.pages)

# Function to use the OpenAI API to answer queries about your data
def query_data(query):
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=f"{data}\n\n{query}",
      temperature=0.5,
      max_tokens=100
    )
    
    # Extract the generated text and print it
    answer = response.choices[0].text.strip()
    print(f"Query: {query}\nAnswer: {answer}")

# Now you can query your data like this:
query = input("Enter your query: ")
query_data(query)