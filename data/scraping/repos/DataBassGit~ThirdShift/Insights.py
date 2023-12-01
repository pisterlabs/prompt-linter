import pprint
import os
import anthropic

def load_csv_as_string(file_path):
    with open(file_path, 'r') as file:
        csv_string = file.read()
    return csv_string

csv_file_path = r'C:\GitKraken\BigBoogaAGI\src\agentforge\llm\MOCK_DATA_SALES.csv'
csv_contents = load_csv_as_string(csv_file_path)

prompt = f"""{anthropic.HUMAN_PROMPT}SystemPrompt:
    As an Insights Generation Agent, your role is to process the provided sales data and generate comprehensive customer profiles that offer valuable insights into customer demographics, preferences, and trends. This information will help sales representatives better understand their customers and improve overall customer service experience.

   
    InstructionPrompt:
    Based on the information provided, please:

    Organize the sales data by AccountNumber.
    Group the data into customer profiles that contain all relevant information for each customer.
    Analyze the customer profiles to identify trends, preferences, and patterns within industries, lead sources, locations, and annual revenue.
    Generate insights that can be used to better understand customers and their behavior.
    Present the insights and customer profiles in an understandable and clear format for use by sales representatives.
    
    
    CSV DATA TO GENERATE CALLBACK LIST WITH:
    {csv_contents}
    
    Response: (Your Response)  
    {anthropic.AI_PROMPT}"""

client = anthropic.Client("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")
response = client.completion(
    prompt=prompt,
    stop_sequences = [anthropic.HUMAN_PROMPT],
    model="claude-v1",
    max_tokens_to_sample=4000,
)
pprint.pprint(response)