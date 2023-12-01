import os
import anthropic
import pprint

def load_csv_as_string(file_path):
    with open(file_path, 'r') as file:
        csv_string = file.read()
    return csv_string

csv_file_path = r'C:\GitKraken\BigBoogaAGI\src\agentforge\llm\SALES_REPS.csv'
csv_contents = load_csv_as_string(csv_file_path)

csv_file_path2 = r'C:\GitKraken\BigBoogaAGI\src\agentforge\llm\MOCK_DATA.csv'
leads = load_csv_as_string(csv_file_path2)

prompt = f"""{anthropic.HUMAN_PROMPT}Variable Info:
    Lead:
    - Potential Profit
    - Industry
    - Owner
    - Filter:IsCold
    - Last Contact

    Sales Rep List:
    - [Sales Rep Name]:
        * Workload
        * Experience
        * Performance

    ---

    SystemPrompt: 

        As a Sales Manager AI specializing in Lead Routing and Assignment, your task is to analyze a given Lead and a list of Sales Reps, then select the Lead that best matches each Sales Rep.

    ---

    

    InstructionPrompt:

	Based on the information provided, please:

	- Analyze the Lead and Sales Reps data.
	- Consider all relevant factors to match the best Lead to each Sales Rep.
	- Prioritize gaining experience for low-experience sales reps with easier or low-profit leads.
	- Match experienced and high-performance reps to more difficult leads or ones with high potential profit.
	- Account for the current workload of sales reps to avoid over-tasking.

	Provide the top 3 choices along with the reasoning behind each choice FOR EACH SALES REP and an overall confidence level. Here is the desired format:

	Confidence: (0-100) (0 being 'No Idea' and 100 being 'Absolute Certainty')

	First Choice: (Top 1 Selected Lead)
	Reason: (Reasoning behind Top 1 selection)

	Second Choice: (Top 2 Selected Lead)
	Reason: (Reasoning behind Top 2 selection)

	Third Choice: (Top 3 Selected Lead)
	Reason: (Reasoning behind Top 3 selection)

    ----

	Here is the relevant information for the task:

	Lead: {leads}

	Sales Rep List: Each Sales Rep comes with the following information: Name, Workload, Experience, and Performance.

	
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