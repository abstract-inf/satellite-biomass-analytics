"""
Created on Sat May 24 20:18:05 2024

@author: Raneem
"""
import numpy as np
import random
import time
import sys

from solution import solution


def crossoverPopulaton(population, scores, popSize, crossoverProbability, keep):
    print("Performing crossover on the population...")
    newPopulation = np.zeros_like(population)
    newPopulation[0:keep] = population[0:keep]

    for i in range(keep, popSize, 2):
        parent1, parent2 = pairSelection(population, scores, popSize)
        individualLength = len(parent1)

        if random.random() < crossoverProbability:
            offspring1, offspring2 = crossover(individualLength, parent1, parent2)
            print(f"Crossover between {parent1} and {parent2} produced {offspring1} and {offspring2}")
        else:
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            print(f"No crossover. Offspring are copies: {offspring1} and {offspring2}")

        newPopulation[i] = offspring1
        newPopulation[i + 1] = offspring2

    print("Crossover completed.")
    return newPopulation


def mutatePopulaton(population, popSize, mutationProbability, keep, lb, ub):
    print("Performing mutation on the population...")
    for i in range(keep, popSize):
        if random.random() < mutationProbability:
            original = population[i].copy()
            mutation(population[i], len(population[i]), lb, ub)
            print(f"Mutation changed {original} to {population[i]}")
    print("Mutation completed.")


def pairSelection(population, scores, popSize):
    print("Selecting pairs of parents...")
    def rouletteWheelSelectionId(inverted_scores, popSize):
        if len(set(inverted_scores)) == 1:
            return random.randint(0, popSize - 1)
        
        total_fitness = sum(inverted_scores)
        normalized_scores = [score / total_fitness for score in inverted_scores]
        cumulative_probs = np.cumsum(normalized_scores)
        random_num = random.random()
        
        for i, cumulative_prob in enumerate(cumulative_probs):
            if random_num <= cumulative_prob:
                return i

    max_score = max(scores)
    inverted_scores = [max_score - score for score in scores]

    parent1Id = rouletteWheelSelectionId(inverted_scores, popSize)
    parent1 = population[parent1Id].copy()

    parent2Id = parent1Id
    while parent2Id == parent1Id:
        parent2Id = rouletteWheelSelectionId(inverted_scores, popSize)
    parent2 = population[parent2Id].copy()

    print(f"Selected parents: {parent1} and {parent2}")
    print("Pair selection completed.")
    return parent1, parent2


def crossover(individualLength, parent1, parent2):
    crossover_point = random.randint(0, individualLength - 1)
    offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return offspring1, offspring2


def mutation(individual, individualLength, lb, ub):
    mutationIndex = random.randint(0, individualLength - 1)
    individual[mutationIndex] = random.uniform(lb[mutationIndex], ub[mutationIndex])


def clearDups(Population, lb, ub):
    print("Clearing duplicates in the population...")
    newPopulation = np.unique(Population, axis=0)
    oldLen = len(Population)
    newLen = len(newPopulation)

    if newLen < oldLen:
        nDuplicates = oldLen - newLen
        randomIndividuals = np.random.uniform(0, 1, (nDuplicates, len(Population[0]))) * (np.array(ub) - np.array(lb)) + np.array(lb)
        newPopulation = np.append(newPopulation, randomIndividuals, axis=0)

    print(f"Cleared {oldLen - newLen} duplicates.")
    print("Duplicates cleared.")
    return newPopulation


def calculateCost(objf, population, popSize, lb, ub):
    print("Calculating fitness values for the population...")
    scores = np.full(popSize, np.inf)

    for i in range(0, popSize):
        population[i] = np.clip(population[i], lb, ub)
        scores[i] = objf(population[i, :])
        print(f"Individual {population[i]} has fitness {scores[i]}")

    print("Fitness calculation completed.")
    return scores


def sortPopulation(population, scores):
    print("Sorting the population based on fitness values...")
    sortedIndices = scores.argsort()
    population = population[sortedIndices]
    scores = scores[sortedIndices]
    print(f"Sorted population: {population}")
    print("Population sorted.")
    return population, scores


def GA(objf, lb, ub, dim, popSize, iters):
    cp = 1
    mp = 0.01
    keep = 2

    s = solution()

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    bestIndividual = np.zeros(dim)
    bestScore = float("inf")

    ga = np.zeros((popSize, dim))
    scores = np.random.uniform(0.0, 1.0, popSize)
    convergence_curve = np.zeros(iters)

    for i in range(dim):
        ga[:, i] = np.random.uniform(0, 1, popSize) * (ub[i] - lb[i]) + lb[i]

    print('GA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(iters):
        print(f"Iteration {l + 1}/{iters}...")
        ga = crossoverPopulaton(ga, scores, popSize, cp, keep)
        mutatePopulaton(ga, popSize, mp, keep, lb, ub)
        ga = clearDups(ga, lb, ub)
        scores = calculateCost(objf, ga, popSize, lb, ub)
        ga, scores = sortPopulation(ga, scores)
        bestIndividual = ga[0]
        bestScore = min(scores)
        convergence_curve[l] = bestScore

        if l % 1 == 0:
            print(f"At iteration {l + 1} the best fitness is {bestScore}")

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart

    s.bestIndividual = bestIndividual
    s.convergence = convergence_curve
    s.optimizer = "GA"
    s.objfname = objf.__name__

    print("GA optimization completed.")
    return s
