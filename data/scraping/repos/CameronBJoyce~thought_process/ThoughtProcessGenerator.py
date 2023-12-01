## Leveraging an LLM for thought process dataset generation
# See bottom for usage.

import pandas as pd
import openai

class ThoughtProcessGenerator:
    def __init__(self, api_key, goal, model="text-davinci-003", max_tokens=100, num_variations=3, temperature=0.7):
        self.api_key = api_key
        self.goal = goal
        self.model = model
        self.max_tokens = max_tokens
        self.num_variations = num_variations
        self.temperature = temperature
        self.results_df = pd.DataFrame(columns=["full prompt", "possible result"])

    def generate_prompt(self, row):
        prompt = f"Goal: {self.goal}\n" # This is the static goal for all prompts you would enter
        for key, value in row.items():
            prompt += f"{key.capitalize()}: {value}\n" # So, Setting: some setting you define in your dataset...etc.
        return prompt

    def generate_thought_processes(self, dataframe):
        openai.api_key = self.api_key 

        for _, row in dataframe.iterrows():
            prompt = self.generate_prompt(row)
            response = openai.Completion.create(
                engine=self.model,
                prompt=prompt,
                max_tokens=self.max_tokens,
                n=self.num_variations,
                temperature=self.temperature,
                stop=None,  # You can set a stop condition here if needed
            )
            
            for choice in response.choices:
                self.results_df = self.results_df.append({"full prompt": prompt, "possible result": choice.text}, ignore_index=True)

        return self.results_df

if __name__ == "__main__":
    api_key = "YOUR_OPENAI_API_KEY" # Substitue me for your own API key! Please don't accidentally push your API key to the repo, SIT!
    
    # Example goal
    goal = "Imagine you are tasked with understanding an individual's thought process as they transition from the start state to the end state described below. Please generate a coherent narrative of their thought process throughout this journey. Consider their emotions, motivations, decisions, and actions as they navigate this transition. Your goal is to generate a thought process narrative that bridges the gap between the start and end states. Provide insights into the person's thoughts, feelings, and decision-making as they progress towards their goal. Be as detailed and realistic as possible."

    
    """
    Ideally you would substitute this for a csv of your own data, but this is just an example set here.
    You can add or drop columns as you want from the data and the code will handle it.
    
    
    Setting (location context for what is happening); Scenario (broad overview); context (additional color); 
    test subject information (personal infrormation on the test subjects); start state (where the story starts); 
    end state (where the story ends)
    """
    dataframe = pd.DataFrame({
        "setting": ["A kitchen set up in a comfortable lab environment", "An book publishing firm that has a harsh corporate environment"],
        "scenario": ["Figuring out what to cook for dinner", "Wanting to write a great work of literature."],
        "context": ["A man is trying to make a good meal to impress on a first date.", "A woman is trying to make is as a successful novelist at her publishing firm."],
        "test subject information": ["Bob is a 22-year-old recent college graduate with a degree in computer science. He's passionate about technology and has just started his first job as a software engineer in a tech startup.", "Linda is a 28-year-old aspiring author who works as an editor at a publishing house. She's determined to write her first novel and is navigating the creative process."],
        "start state": ["An individual with ingredients in hand but unsure of what to cook.", "Linda has a manuscript but wants to be taken seriously."],
        "end state": ["Successfully preparing a delicious meal.", "Linda decides to write a cold email to her boss's boss letting them know of her work."]
    })
    # dataframe = pd.read_csv('SIT's_awesome_prompt_data.csv')
    
    generator = ThoughtProcessGenerator(api_key, goal) #change num_variations to change the number of responses for each prompt
    result_df = generator.generate_thought_processes(dataframe)

    print(result_df)
