#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:22:00 2023

@author: Michael Glass
"""

import numpy as np
import random
from random import randint
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, squareform

data=pd.read_csv('world_coordinates.csv').drop("Code",axis=1)
df=pd.DataFrame(data).set_index("Country")
x=df[["longitude","latitude"]].to_numpy()
Distances = pdist(x, metric='euclidean')
Distance_matrix = squareform(Distances)


# Number of cities in TSP
numberOfCities = 4

# Names of the cities
GENES = "BLMS"

# Starting Node Value
START = 0

# Genetic algorithm parameters
population_size = 10
generations = 10
mutation_rate = 0.1

# Function to return a random number
# from start and end
def rand_num(start, end):
	return np.array([randint(start, end-1)])


# Function to check if the character
# has already occurred in the string
def repeat(s, num):
	for i in range(len(s)):
		if s[i] == num:
			return True

	return False

def generate_initial_population(population_size):
    population = []
    for i in range(population_size):
        # Create a random permutation of cities
        gnome=np.array([0])
        while True:
            if len(gnome) == numberOfCities:
                gnome = np.concatenate((gnome, np.array([gnome[0]]))) 
                break

            temp =rand_num(1, numberOfCities)
            if not repeat(gnome, temp): 
                gnome =np.concatenate((gnome,temp))
            
        population.append(gnome)
    return population

def calculate_fitness(route, distances):
    total_distance = 0
    for i in range(len(route) - 1):
        city1 = route[i]
        city2 = route[i + 1]
        distance = distances[city1, city2]
        total_distance += distance
    return total_distance

def selection(population, fitnesses):
    # Sort population and fitnesses together
    sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1])
    selected_population = []
    for i in range(len(population) // 2):
        selected_population.append(sorted_population[i][0])
        selected_population.append(sorted_population[len(population) - 1 - i][0])
    return selected_population

def crossover(parents, offspring_size):
    offspring = []
    for i in range(0, offspring_size, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]

        # Create offspring by swapping segments between parents
        crossover_point = random.randint(1, len(parent1) - 2)
        offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        
        
        
        offspring.append(offspring1)
        offspring.append(offspring2)
    return offspring


def mutation(offspring, mutation_rate):
    for i in range(len(offspring)):
        # Randomly mutate genes with probability mutation_rate
    
        for j in range(1,numberOfCities):
            if random.random() < mutation_rate:
                mutation_point1 = random.randint(1,numberOfCities-1) 
                mutation_point2 = random.randint(1,numberOfCities-1)
                if mutation_point1 != mutation_point2:
                    temp = offspring[i][mutation_point1]
                    offspring[i][mutation_point1] = offspring[i][mutation_point2]
                    offspring[i][mutation_point2] = temp
    return offspring

def genetic_algorithm(distances, population_size, generations, mutation_rate):
    # Initialize population
    population = generate_initial_population(population_size)

    # Iterate for a number of generations
    for i in range(generations):
        # Calculate fitness of each individual
        fitnesses = []
       
        for route in population:
            fitness = calculate_fitness(route, distances)
            fitnesses.append(fitness)
            

        # Select individuals for reproduction
        selected_population = selection(population, fitnesses)

        # Generate offspring through crossover
        offspring = crossover(selected_population, population_size)

        # Apply mutation
        mutated_offspring = mutation(offspring, mutation_rate)

        # Combine new offspring with selected parents to form the next generation
        new_population = selected_population + mutated_offspring

        # Replace current population with new generation
        population = new_population

    # Find the best individual in the final population

    best_route = population[fitnesses.index(min(fitnesses))]
    best_fitness = fitnesses[fitnesses.index(min(fitnesses))]

    return best_route, best_fitness

if __name__ == "__main__":
    # Example usage with a small TSP instance
    distances = np.array([
        [0, 3, 7.6, 7.8],
        [3, 0, 4.5, 5.7],
        [7.6, 4.5, 0, 3.1],
        [7.8, 5.7, 3.1, 0]
    ])

    # Genetic algorithm parameters
    population_size = 10
    generations = 10
    mutation_rate = 0.1

    # Run the genetic algorithm
    best_route, best_fitness = genetic_algorithm(distances, population_size, generations, mutation_rate)
    print(best_route,best_fitness)    
    
    #
    numberOfCities = len(data.Country)
    best_route, best_fitness = genetic_algorithm(Distance_matrix, population_size, generations, mutation_rate)
    print(best_route,best_fitness)  
    

