import os
import anthropic

def load_csv_as_string(file_path):
    with open(file_path, 'r') as file:
        csv_string = file.read()
    return csv_string

csv_file_path = r'C:\GitKraken\BigBoogaAGI\src\agentforge\llm\MOCK_DATA.csv'
csv_contents = load_csv_as_string(csv_file_path)

prompt = f"""{anthropic.HUMAN_PROMPT}Variable Info:
    
    Cold Lead Data Extractor Agent Output:

    Lead Data

    Lead Prioritization Agent Output:

    Prioritized Leads

    Rep Matchmaking Agent Output:

    Matched Representatives

    Contact Strategy Agent Output:

    Recommended Contact Strategies

    Coaching and Follow-up Strategy Agent Output:

    Coaching Tips
    Follow-up Strategies

    SystemPrompt:

    As a Call List Generation Agent, your task is to compile an optimized call list based on the provided prioritized leads, matched representatives, and recommended contact strategies received from prior agents.

    InstructionPrompt:

    Based on the information provided, please:

    Organize the call list starting with the highest-priority leads.
    Include the matched sales representative for each lead.
    Incorporate the recommended contact strategy for each lead.
    Provide context-sensitive coaching tips and follow-up strategies for each sales representative per lead.
    Format the call list as follows:
    
    Call List:
    
    Lead: 
    Sales Representative: 
    Contact Strategy: 
    Coaching Tips: 
    Follow-up Strategies: 
    
    Lead: 
    Sales Representative: 
    Contact Strategy: 
    Coaching Tips: 
    Follow-up Strategies: 
    
    (Repeat in the same format for all leads)
    
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
print(response)