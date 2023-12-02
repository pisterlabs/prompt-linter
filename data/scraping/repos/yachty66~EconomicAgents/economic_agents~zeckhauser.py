from collections import Counter
import logging 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .open_ai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Zeckhauser:
    def __init__(self, api_key, model, image_path, logging):
        self.model = OpenAI(key=api_key, model=model)
        self.image_path = image_path
        self.logging = logging
        self.views = [
            'car safety is the most important thing.',
            'car safety is a terrible waste of money; we should only fund highway safety.',
            'car safety is all that matters. We should not fund highway safety.',
            'car safety and highway safety are equally important',
            'car safety is slightly more important than highway safety',
            "I don't really care about car safety or highway safety",
            'highway safety is the most important thing.',
            'highway safety is a terrible waste of money; we should only fund car safety.',
            'highway safety is all that matters. We should not fund car safety.',
            'highway safety and car safety are equally important',
            'highway safety is slightly more important than car safety',
            "I don't really care about highway safety or car safety"
        ]
        self.status_quo = [30, 50, 60, 70, " "]
        self.options = ['a', 'b', 'c', 'd']

    def play(self):
        generate_status_quo = [
            "The current budget allocation is 30% to auto safety and 70% to highway safety.",
            "The current budget allocation is 50% to auto safety and 50% to highway safety.", 
            "The current budget allocation is 60% to auto safety and 40% to highway safety.",
            "The current budget allocation is 70% to auto safety and 30% to highway safety.",
            " "
        ]
        results = {}
        for element in generate_status_quo:
            for view in self.views:
                prompt = f"""The National Highway Safety Commission is deciding how to allocate its budget between two safety research programs: 

                i) improving automobile safety (bumpers, body, gas tank configurations, seatbelts) and 
                ii) improving the safety of interstate highways (guard rails, grading, highway interchanges, and implementing selective reduced speed limits).

                {element}

                They are considering the following options:

                a) Allocate 70% to auto safety and 30% to highway safety
                b) Allocate 40% to auto safety and 60% to highway safety
                c) Allocate 30% to auto safety and 70% to highway safety
                d) Allocate 50% to auto safety and 50% to highway safety

                Your own views: {view}

                Please choose your most preferred option in light of your views a, b, c or d. Its important that you respond only with one letter and not more!:"""
                solution = self.model.generate(prompt)
                results[(view, element)] = solution
                if self.logging:
                    logger.info(f"Prompt: {prompt}")
                    logger.info(f"Model response: {solution}")
        return results
    
    def calculate_percentages(self,counter):
        total = sum(counter.values())
        return {key: (value / total) * 100 for key, value in counter.items()}
        
    def create_plot(self, results):
        status_quo_30 = []
        status_quo_50 = []
        status_quo_60 = []
        status_quo_70 = []
        neutral_status_quo = []
        for key, value in results.items():
            second_element = key[1]
            first_digit = next((char for char in second_element if char.isdigit()), None)
            if first_digit == '3':
                status_quo_30.append({key: value})
            elif first_digit == '5':
                status_quo_50.append({key: value})
            elif first_digit == '6':
                status_quo_60.append({key: value})
            elif first_digit == '7':
                status_quo_70.append({key: value})
            else:
                neutral_status_quo.append({key: value})
        status_quo_30_data = self.calculate_percentages(Counter([list(d.values())[0] for d in status_quo_30]))
        status_quo_50_data = self.calculate_percentages(Counter([list(d.values())[0] for d in status_quo_50]))
        status_quo_60_data = self.calculate_percentages(Counter([list(d.values())[0] for d in status_quo_60]))
        status_quo_70_data = self.calculate_percentages(Counter([list(d.values())[0] for d in status_quo_70]))
        neutral_status_quo_data = self.calculate_percentages(Counter([list(d.values())[0] for d in neutral_status_quo]))
        views = ["30% auto \n framed as Status Quo", "50% auto \n framed as Status Quo", "60% auto \n framed as Status Quo", "70% auto \n framed as Status Quo", "Neutral framing"]
        choices = ["70% car, 30% hwy", "40% car, 60% hwy", "30% car, 70% hwy", "50% car, 50% hwy"]
        fig, axs = plt.subplots(1, 5, figsize=(20, 3.5), sharey=True)
        data_list = [status_quo_30_data, status_quo_50_data, status_quo_60_data, status_quo_70_data, neutral_status_quo_data]
        for i, ax in enumerate(axs):
            keys = list(data_list[i].keys())
            values = list(data_list[i].values())
            ax.bar(np.arange(len(keys)), values) 
            ax.set_title(views[i])
            ax.set_xticks(np.arange(len(choices)))  
            ax.set_xticklabels(choices, rotation='vertical') 
            ax.set_ylim(0, 100)  
            ax.set_yticks([0, 20, 40, 60, 80]) 
            ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%'])  
        plt.tight_layout()
        plt.savefig(self.image_path)
        if self.logging:
            logger.info(f"Plot saved at: {self.image_path}")
        plt.show()

    def __call__(self):
        results = self.play()
        self.create_plot(results)
        return results