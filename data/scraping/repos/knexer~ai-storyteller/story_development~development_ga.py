import openai

from ga.ga import GeneticAlgorithmBase
from ga.individual import Individual
from outline_story import Outliner
from story_development.characters import Characters
from story_development.outline import Outline
from story_development.setting import Setting
from story_development.themes import Themes


class DevelopmentGA(GeneticAlgorithmBase):
    def __init__(self, conditioning_info, premise, inital_ideas):
        self.conditioning_info = conditioning_info
        self.premise = premise
        initial_population = [self.make_individual(idea) for idea in inital_ideas]
        GeneticAlgorithmBase.__init__(self, initial_population)

    def make_individual(self, notes):
        return Individual(
            [
                Characters(self.conditioning_info, self.premise, notes),
                Setting(self.conditioning_info, self.premise, notes),
                Themes(self.conditioning_info, self.premise, notes),
                Outline(self.conditioning_info, self.premise, notes),
            ]
        )

    def compute_fitness(self, individual):
        if not individual.is_scored():
            individual.score()

        return individual.total_score()

    def mutate(self, individual):
        category, recommendation = individual.make_recommendation()
        print(f"Got recommendation for {category.category_name()}: {recommendation}")
        mutated_notes = individual.apply_recommendation(category, recommendation)
        print(f"Revised notes: {mutated_notes}")
        return self.make_individual(mutated_notes)

    def crossover(self, parent1, parent2):
        # Given two individuals, parent1 as the primary and parent2 as the secondary
        # Identify the best things about parent2, based on the feedback
        # Update parent1 to incorporate those best things
        raise NotImplementedError(
            "Derived classes must implement the crossover operator"
        )
