import openai
import os
import csv
import login

#INSIGHTS ***********************************************
propertyAddress = []
insight1 = []
insight2 = []
driver = []
account = []
criticality = []

with open('property_insights_extended.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row if it exists
    for row in reader:
        propertyAddress.append(row[0])
        insight1.append(row[1])
        insight2.append(row[2])  
        driver.append(row[3])  
        account.append(row[4])
        criticality.append(row[5])

'''
def GPT():
    # Set your OpenAI API key here
    openai.api_key = os.getenv("OPENAI_API_KEY", "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")

    # Specify file path for csv reading
    csv_file_path = 'property_insights_extended.csv'

    # Open the CSV file
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        second_row = next(csv_reader)

    # Define the user prompt message
    string = ', '.join(second_row)
    prompt = "Make a sentence: " + string

    # Create a chatbot using ChatCompletion.create() function
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    insights = [choice.message for choice in completion.choices]

    return insights
'''

def promptGPT(index):
    # Set your OpenAI API key here
    openai.api_key = os.getenv("OPENAI_API_KEY", "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")
    string = propertyAddress[index] + ", " + insight1[index] + ", " + insight2[index] + ", " + driver[index]
    prompt = "Make an insight sentence for a real estate app: " + string
    #print("Prompt: " + prompt)
    
    messages = [
        {"role": "assistant", "content": prompt}
    ]
    
    # Make the API call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Get the generated sentence from the response
    insights = response['choices'][0]['message']['content']

    return insights

def getAccount(index):
    return account[index]

def getPriorities():
    return insight1

def getCriticality():
    return criticality

def getInsight2(index):
    return insight2[index]