import openai
import re
from googlesearch import search
import re

# Set OpenAI API credentials
openai.api_key = "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"

# Define the sentence to fact-check
sentence = "The capital of France is London."

# Extract the claim to fact-check
match = re.search(r"[A-Za-z]+ (is|was|are|were) [A-Za-z ]+\.", sentence)
if match:
    claim = match.group(0)
else:
    print("No claim found in the sentence.")
    exit()

# Use OpenAI to check the claim
result = openai.Completion.create(
    engine="davinci",
    prompt=f"Is the following claim true or false?\nClaim: {claim}\nTrue\nFalse",
    max_tokens=1,
    n=1,
    stop=None,
    temperature=0.7,
)

# Print the fact-check result
if result.choices[0].text == "True":
    print("The claim is true.")
elif result.choices[0].text == "False":
    print("The claim is false.")
else:
    print("Unable to determine the truth value of the claim.")



# Define the sentence to fact-check
sentence = "The capital of France is London."

# Extract the claim to fact-check
match = re.search(r"[A-Za-z]+ (is|was|are|were) [A-Za-z ]+\.", sentence)
if match:
    claim = match.group(0)
else:
    print("No claim found in the sentence.")
    exit()

# Search Google for relevant information
query = f"{claim} site:wikipedia.org"
results = list(search(query, num=5, stop=5))

# Process the search results to determine the fact-check result
for result in results:
    if "wikipedia.org" in result:
        response = requests.get(result)
        text = response.text
        if "redirect" in response.url:
            redirect_url = re.search(r"url=([^&]+)", response.url).group(1)
            response = requests.get(redirect_url)
            text = response.text
        if "not exist" in text or "does not have an article" in text:
            continue
        if "not be verified" in text or "cannot be confirmed" in text:
            print("Unable to determine the truth value of the claim.")
            break
        if "true" in text.lower():
            print("The claim is true.")
            break
        elif "false" in text.lower():
            print("The claim is false.")
            break
else:
    print("Unable to determine the truth value of the claim.")

