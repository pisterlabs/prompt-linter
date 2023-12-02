import openai
import re
import json
import tenacity

# define your openai api key
# openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = "sk-..."

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
def eliminate(model=None):
    # Load the JSON file
    with open('pseudo_competitors.json', 'r') as f:
        data = json.load(f)

    # num_runs is the length of the JSON file
    num_runs = len(data)

    # open user_instructions.txt and read the contents
    with open('user_instructions.txt', 'r') as f:
        user_instructions = f.read()

    candidate = data['1']

    for run in range(num_runs):

        # Initialize the API call
        response = openai.ChatCompletion.create(
            model=model,
            temperature=0,
                messages=[
                {"role": "system", "content": "You are a very critical expert high level outline evaluator."},
                {"role": "user", "content": f"""
                candidate: 
                
                {candidate}

            approve or disapprove the candidate high level outline with an extremely careful programer's scepticism following these criterias in order of importance:
                
                1- Best high level outline implements all of the user specifications and instructions
                2- Best high level outline has an excellent programmatic code structure which will produce efficient and effective code
                3- Best high level outline needs to be able to lead to readable, clean code

                user instructions: 

                {user_instructions}

                
                give an answer only as approved or disapproved and do not return anything else other than the conclusion:
                always include the word "conclusion" in your answer:

                EXAMPLES OF ANSWER FORMAT:           
                conclusion = "approved" 
                conclusion = "disapproved"
                """}
            ]
        )
        # Attention!: give concise keyword reasons and answer only as and do not return anything else other than the reasons and winner:
        #         reasons = "reason 1: keyword, keyword, keyword" and 
        #                 "reason 2: keyword, keyword, keyword" and 
        #                 "reason 3: keyword, keyword, keyword"
        #         winner = "high level outline 1" or "high level outline 2'''

        # reasons:
        #         reason 1: .....
        #         reason 2: .....
        #         reason 3: .....

        print(response['choices'][0]['message']['content'])
        winning_pseudo_code = response['choices'][0]['message']['content']
        if winning_pseudo_code == 'conclusion = approved' :
            winning_pseudo_code = 'conclusion = "approved"'
        elif winning_pseudo_code == 'conclusion = disapproved' :
            winning_pseudo_code = 'conclusion = "disapproved"'

        status_pseudo_code = re.findall(r'conclusion = "(.*?)"', winning_pseudo_code, re.DOTALL)
        
        print(status_pseudo_code)

        # remove the high level outline that was disapproved
        if len(status_pseudo_code):
            if status_pseudo_code[0] == 'disapproved':
                data.pop('1')
        elif status_pseudo_code == "disapproved":
            data.pop('1')

        # reorder the JSON objects elements starting from 1 using enumerate
        new_data = {}
        for i, (key, value) in enumerate(data.items(), start=1):
            new_data[str(i)] = value
        
        data = new_data

    # write the remaining high level outlines to a new JSON file
    with open('winning_pseudo_competitors.json', 'w') as f:
        json.dump(data, f)

    # write the remaining high level outlines to separate files
    # for key, value in data.items():
    #     with open(f'pseudo_competitors_{key}.py', 'w') as f:
    #         f.write(value)

if __name__ == '__main__':
    eliminate(model="gpt-4")

