import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

training = [["First Name", "Medium"], ["Last Name", "Medium"], ["IP Address", "Medium"], ["Address", "Medium"], ["City", "Medium"], ["State", "Medium"], 
    ["Zip Code", "Medium"], ["Phone Number", "Medium"], ["Email Address", "Medium"], ["Date of Birth", "High"], ["SSN", "Secret"], 
    ["Credit Card Number", "Secret"], ["Credit Card Expiration Date", "Secret"], ["Credit Card CVV", "Secret"], ["Bank Account Number", "High"], 
    ["Driver's License Number", "High"], ["Driver's License State", "High"], ["Driver's License Expiration Date", "High"], ["Passport Number", "High"],
    ["Passport Country", "High"], ["Passport Expiration Date", "High"], ["Mother's Maiden Name", "Medium"], ["Mother's Birth City", "Medium"],
    ["Bank Routing Number", "Medium"], ["Bank Account Type", "Medium"], ["Bank Account Balance", "Medium"], ["Bank Account Currency", "Medium"], 
    ["Bank Account Country", "Medium"], ["Bank Account Owner", "Medium"], ["Bank Account Owner Address", "Medium"], ["Bank Account Owner City", "Medium"], 
    ["Bank Account Owner State", "Medium"], ["Bank Account Owner Zip Code", "Medium"], ["Bank Account Owner Phone Number", "Medium"], 
    ["Bank Account Owner Email Address", "Medium"], ["Bank Account Owner Date of Birth", "Medium"], ["Bank Account Owner SSN", "Secret"], 
    ["Bank Account Owner Credit Card Number", "Secret"], ["Bank Account Owner Credit Card Expiration Date", "Secret"], ["Bank Account Owner Credit Card CVV", "Secret"], 
    ["Bank Account Owner Bank Account Number", "High"], ["Bank Account Owner Bank Routing Number", "Medium"], ["Bank Account Owner Bank Account Type", "Medium"], 
    ["Bank Account Owner Bank Account Balance", "Medium"], ["Bank Account Owner Bank Account Currency", "Medium"], ["Bank Account Owner Bank Account Country", "Medium"]]

# take in a query via command line
query = input("Enter a query: ")

# Convert training data to text format for the generative model
training_text = "\n".join([f"{example[0]}: {example[1]}" for example in training])

# Create a prompt for the generative model
prompt = f"Given the following training data:\n{training_text}\n\nDetermine the data risk classification level of the query: '{query}'\n"

completion = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=10,
    n=1,
    stop=None,
    temperature=0.5,
)

prediction = completion.choices[0].text.strip().lower()

print("The classification is: " + prediction)
