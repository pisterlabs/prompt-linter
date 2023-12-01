import matplotlib.pyplot as plt
import logging 

from .open_ai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Horton:
    def __init__(self, api_key, model, image_path, logging):
        self.openai_api_key = api_key
        self.model = OpenAI(key=api_key, model=model)
        self.image_path = image_path
        self.logging = logging

    def play(self):
        wage_asks = [13, 14, 15, 16, 17, 18, 19, 20]
        min_wages = [13, 15]
        results = {}
        for i in min_wages:
             for j in wage_asks:
                prompt = f"""
                You are hiring for the role "Dishwasher."

                The typical hourly rate for this role is $12/hour.

                You have 2 candidates.

                Person 1: Has 1 year(s) of experience in this role. Requests ${j}/hour.
                Person 2: Has 0 year(s) of experience in this role. Requests ${i}/hour.

                Who would you hire? You have to pick one. Its important that you only answer with the number and nothing else:
                """
                solution = self.model.generate(prompt)
                scenario = f"Min wage: {i}, Wage ask: {j}"
                results[scenario] = solution
                if self.logging:
                    logger.info(f"Prompt: {prompt}")
                    logger.info(f"Model response: {solution}")
        return results
    
    def create_plot(self, results):
        scenarios = list(results.keys())
        choices = [int(choice) for choice in results.values()]
        plt.figure(figsize=(12, 6))
        plt.bar(scenarios, choices, color='blue')
        plt.xlabel('Scenarios')
        plt.ylabel('Choices (1 for Person 1, 2 for Person 2)')
        plt.title('Hiring choice for each scenario')
        plt.xticks(rotation=90)  
        plt.savefig(self.image_path)

    def __call__(self):
        results = self.play()
        self.create_plot(results)
        return results
