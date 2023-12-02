"""
This tool uses a genetic algorithm to tune the hyperparameters of the GPT-3 model.
It uses two methods to evaluate the fitness of a candidate hyperparameter set: 
(1) the perplexity of the generated text and 
(2) the cosine similarity between the generated text and the target response.

The fitness function is a weighted sum of the two methods. The tools invokes the algorithm with 
different weights to find the optimal hyperparameters. The final judgement of the optimal
hyperparameters is based on the human judgement of the generated text.
"""
import random
import json
from sklearn.metrics.pairwise import cosine_similarity
from .openaicli import OpenAICli

with open('training_data.json', 'r', encoding='utf-8') as file:
    training_data = json.load(file)

oac = OpenAICli()

# Define the hyperparameters
temperature_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
top_p_range = [0.5, 0.7, 0.9, 1]
frequency_penalty_range = [0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8, 2]
presence_penalty_range = [0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 1.8, 2]


def fitness(hyperparameters, messages, target_response, p_w, s_w):
    """
    Define the fitness function
    """
    generated_text, perplexity = oac.get_response(messages, hyperparameters)

    generated_vector = oac.get_embedding(generated_text)
    target_vector = oac.get_embedding(target_response)
    similarity = cosine_similarity(
        generated_vector.reshape(1, -1), target_vector.reshape(1, -1))
    score = p_w * \
        (1.0 / float(perplexity)) + s_w * similarity[0][0]

    return score


def generate_individual():
    """
    Generate a random individual
    """
    temperature = random.choice(temperature_range)
    top_p = random.choice(top_p_range)
    frequency_penalty = random.choice(frequency_penalty_range)
    presence_penalty = random.choice(presence_penalty_range)

    return {'temperature': temperature, 'top_p': top_p, 'frequency_penalty': \
            frequency_penalty, 'presence_penalty': presence_penalty}


def generate_population(population_size=100):
    """
    Generate a population of random individuals
    """
    population = [generate_individual() for _ in range(population_size)]

    return population


def evaluate_population(population, p_w, s_w):
    """
    Evaluate the fitness of each individual in the population
    """
    fitness_scores = []
    for individual in population:
        prompt = random.choice(training_data['prompt'])
        target_response = training_data.loc[training_data['prompt']
                                            == prompt, 'response'].values[0]
        score = fitness(individual, prompt, target_response, p_w, s_w)
        fitness_scores.append(score)

    return fitness_scores


def alpha(population, population_size, alpha_parent):
    """
    Breed the population by having the alpha seed the next generation
    """
    new_population = []
    while len(new_population) < population_size:
        parent = random.choice(population)
        child = {}
        for key in parent.keys():
            if random.random() < 0.5:
                child[key] = parent[key]
            else:
                child[key] = alpha_parent[key]
        new_population.append(child)

    return new_population


def crossover(population, population_size):
    """
    Breed the population by random crossover
    """
    new_population = []
    while len(new_population) < population_size:
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        new_population.append(child)

    return new_population


def breed_popuplation(population, population_size, strategy, alpha_parent):
    """
    Breed the population using the specified strategy
    """
    if strategy == 'crossover':
        return crossover(population, population_size)
    elif strategy == 'alpha':
        return alpha(population, population_size, alpha_parent)
    else:
        raise ValueError('Unknown strategy')


def mutate_population(population):
    """
    Mutate the population
    """
    for individual in population:
        if random.random() < 0.1:
            individual['temperature'] = random.choice(temperature_range)
        if random.random() < 0.1:
            individual['top_p'] = random.choice(top_p_range)
        if random.random() < 0.1:
            individual['frequency_penalty'] = random.choice(
                frequency_penalty_range)
        if random.random() < 0.1:
            individual['presence_penalty'] = random.choice(
                presence_penalty_range)

    return population


def evolve_population(strategy, population, population_size, fitness_scores, alpha_parent):
    """
    Evolve the population
    """
    num_selected = int(population_size * 0.2)
    selected_indices = sorted(range(
        population_size), key=lambda i: fitness_scores[i], reverse=True)[:num_selected]
    selected_population = [population[i] for i in selected_indices]

    new_population = breed_popuplation(
        selected_population, population_size, strategy, alpha_parent)
    new_population = mutate_population(new_population)

    return new_population


def run_genetic_algorithm(strategy, population_size, num_generations, p_w, s_w):
    """
    Run the genetic algorithm
    """
    population = generate_population(population_size)
    fitness_scores = evaluate_population(population, p_w, s_w)
    alpha_parent = population[fitness_scores.index(max(fitness_scores))]

    # Evolution loop
    num_generations = 10
    for generation in range(num_generations):
        print('Generation:', generation)
        new_population = evolve_population(strategy,
            population, population_size, fitness_scores, alpha_parent)

        # Evaluate the fitness of the new population
        new_fitness_scores = evaluate_population(new_population, p_w, s_w)

        # Replace the old population with the new population
        population = new_population
        fitness_scores = new_fitness_scores

        alpha_parent = population[fitness_scores.index(max(fitness_scores))]

        print(f"Generation:{generation}, Best individual:{alpha_parent}")

    return alpha_parent

def run_with_strategy(strategy, population_size, generations):
    """
    Run the genetic algorithm with the specified strategy
    """
    weights = {(0, 1), (1, 0), (0.25, 0.75), (0.75, 0.25), (0.5, 0.5)}

    for weight in weights:
        perplexity_weight, similarity_weight = weight
        best_individual = run_genetic_algorithm(strategy,
            population_size, generations, perplexity_weight, similarity_weight)
        print(f'perplexity_weight:{perplexity_weight},\
              similarity_weight:{similarity_weight},\
              Best individual:{best_individual}')

if __name__ == '__main__':
    run_with_strategy('crossover', 100, 10)
    run_with_strategy('alpha', 100, 10)
