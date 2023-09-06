__author__ = "Ben Knox"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "knobe957@student.otago.ac.nz"

import numpy as np
import random

agentName = "<my_agent>"
# Train against random agent for 5 generations,
trainingSchedule = [("random_agent.py", 5), ("self", 1)]
# then against self for 1 generation

# This is the class for your cleaner/agent


class Cleaner:

    def __init__(self, nPercepts, nActions, gridSize, maxTurns):
        # This is where agent initialisation code goes (including setting up a chromosome with random values)

        # Leave these variables as they are, even if you don't use them in your AgentFunction - they are
        # needed for initialisation of children Cleaners.
        self.nPercepts = nPercepts
        self.nActions = nActions
        self.gridSize = gridSize
        self.maxTurns = maxTurns

        # could be in the wrong place, might want it in agent function
        self.chromosome = self.createInitialChromosome()

    def createInitialChromosome(self):
        # # Will give negative weightings, not sure if I want this yet
        # energyRand = np.random.uniform(-1, 1)
        # binRand = np.random.uniform(-1, 1)
        # cleanRand = np.random.uniform(-1, 1)
        # collisionAvoidanceRand = np.random.uniform(-1, 1)

        # # energyRand = np.random(100)
        # # binRand = np.random(100)
        # # cleanRand = np.random(100)
        # # collisionAvoidanceRand = np.random(100)

        # sumRand = abs(energyRand) + abs(binRand) + \
        #     abs(cleanRand) + abs(collisionAvoidanceRand)

        # energyWeight = energyRand / sumRand
        # binWeight = binRand / sumRand
        # cleanWeight = cleanRand / sumRand
        # collisionAvoidanceWeight = collisionAvoidanceRand / sumRand

        # chromosome = [energyWeight, binWeight,
        #               cleanWeight, collisionAvoidanceWeight]
        self.chromosomeDict = {}

        for j in range(4):
            chromosome = np.random.uniform(-1, 1, size=63)
            bias = np.random.uniform(-1, 1)
            # chromosome = np.random.uniform(0, 1, size=63)
            # bias = np.random.uniform(0, 1)
            full_chromosome = np.concatenate((chromosome, [bias]))
            sum_chromosome = np.sum(full_chromosome)
            self.chromosomeDict[sum_chromosome] = full_chromosome

        chromosome = self.chromosomeDict[max(self.chromosomeDict.keys())]

        return chromosome

    def AgentFunction(self, percepts):
        # preferredActionIndex = np.searchsorted(cumulativeWeights, randomNumber)
        # selected_value = chromosome[selected_index]

        # The percepts are a tuple consisting of four pieces of information
        #
        # visual - it information of the 3x5 grid of the squares in front and to the side of the cleaner; this variable
        #          is a 3x5x4 tensor, giving four maps with different information
        #          - the dirty,clean squares
        #          - the energy
        #          - the friendly and enemy cleaners that are able to traverse vertically
        #          - the friendly and enemy cleaners that are able to traverse horizontally
        #
        #  energy - int value giving the battery state of the cleaner -- it's effectively the number of actions
        #           the cleaner can still perform before it runs out of charge
        #
        #  bin    - number of free spots in the bin - when 0, there is no more room in the bin - must emtpy
        #
        #  fails - number of consecutive turns that the agent's action failed (rotations always successful, forward or
        #          backward movement might fail if it would result in a collision with another robot); fails=0 means
        #          the last action succeeded.

        visual, energy, bin, fails = percepts

        # You can further break down the visual information

        # 3x5 map where -1 indicates dirty square, 0 clean one
        floor_state = visual[:, :, 0]
        flat_floor_state = floor_state.flatten()

        # 3x5 map where 1 indicates the location of energy station, 0 otherwise
        energy_locations = visual[:, :, 1]
        flat_energy_locations = energy_locations.flatten()

        # 3x5 map of bots that can in this turn move up or down (from this bot's point of
        vertical_bots = visual[:, :, 2]
        flat_vertical_bots = vertical_bots.flatten()
        # view), -1 if the bot is an enemy, 1 if it is friendly

        # 3x5 map of bots that can in this turn move up or down (from this bot's point
        horizontal_bots = visual[:, :, 3]
        flat_horizontal_bots = horizontal_bots.flatten()
        # of view), -1 if the bot is an enemy, 1 if it is friendly

        self.flattenedVisuals = np.concatenate(
            (flat_floor_state, flat_energy_locations, flat_vertical_bots, flat_horizontal_bots))
        self.flattenedVisuals = np.concatenate(
            (self.flattenedVisuals, [energy, bin, fails]))

        # You may combine floor_state and energy_locations if you'd like: floor_state + energy_locations would give you
        floor_plus_energy = floor_state + energy_locations
        # a map where -1 indicates dirty square, 0 a clean one, and 1 an energy station.

        # You should implement a model here that translates from 'percepts' to 'actions'
        # through 'self.chromosome'.

        #
        # The 'actions' variable must be returned, and it must be a 4-item list or a 4-dim numpy vector

        # The index of the largest value in the 'actions' vector/list is the action to be taken,
        # with the following interpretation:
        # largest value at index 0 - move forward;
        # largest value at index 1 - turn right;
        # largest value at index 2 - turn left;
        # largest value at index 3 - move backwards;
        #
        # Different 'percepts' values should lead to different 'actions'.  This way the agent
        # reacts differently to different situations.
        #
        # Different 'self.chromosome' should lead to different 'actions'.  This way different
        # agents can exhibit different behaviour.

        action_vector = list(self.chromosomeDict.keys())

        action_vector = np.zeros(4)  # Initialize action_vector with zeros
        i = 0

        flattenedWithBias = np.concatenate((self.flattenedVisuals, [1]))

        # Iterate through the values in the dictionary
        for values in self.chromosomeDict.values():
            # Perform element-wise multiplication with self.flattenedVisuals
            new_values = np.array(flattenedWithBias) * np.array(values)

            # Sum the new_values and store it in the action_vector
            action_vector[i] = np.sum(new_values)
            i += 1

        # Right now this agent ignores percepts and chooses a random action.  Your agents should not
        # perform random actions - your agents' actions should be deterministic from
        # computation based on self.chromosome and percepts

        # action_vector = np.random.randint(
        #     low=-100, high=100, size=self.nActions)
        return action_vector


def evalFitness(population):

    N = len(population)

    # Fitness initialiser for all agents
    fitness = np.zeros((N))

    # This loop iterates over your agents in the old population - the purpose of this boilerplate
    # code is to demonstrate how to fetch information from the old_population in order
    # to score fitness of each agent

    for n, cleaner in enumerate(population):
        stats = cleaner.game_stats

        # Objective 1: Maximize the number of cleaned squares
        cleaned_squares = stats['cleaned']

        # Objective 2: Minimize energy consumption (reward energy efficiency)
        # Energy gained from charging stations
        energy_consumed = stats['recharge_energy']
        # Total turns with non-zero energy
        active_turns = stats['active_turns']

        same_square = stats['visits']

        # Objective 3: Encourage bin emptying
        emptied_bins = stats['emptied']

        # You can define weights to balance the importance of these objectives
        weight_cleaned_squares = 1.0
        weight_emptied_bins = 0.5
        weight_same_square = 0.1
        weight_active_turns = 0.2

        fitness[n] = (
            weight_cleaned_squares * cleaned_squares +
            weight_emptied_bins * emptied_bins +
            weight_same_square * same_square +
            weight_active_turns * active_turns
        )

    # sum_fitness = sum(fitness)
    # normalized_fitness = [fit / sum_fitness for fit in fitness]
    # cumulative_distribution = np.cumsum(normalized_fitness)
    # print("CUMSUM RESULT::: ", cumulative_distribution)
    # fitness = cumulative_distribution

    # for n, cleaner in enumerate(population):
    #     # cleaner is an instance of the Cleaner class that you implemented above, therefore you can access any attributes
    #     # (such as `self.chromosome').  Additionally, each object have 'game_stats' attribute provided by the
    #     # game engine, which is a dictionary with the following information on the performance of the cleaner in
    #     # the last game:
    #     #
    #     #  cleaner.game_stats['cleaned'] - int, total number of dirt loads picked up
    #     #  cleaner.game_stats['emptied'] - int, total number of dirt loads emptied at a charge station
    #     #  cleaner.game_stats['active_turns'] - int, total number of turns the bot was active (non-zero energy)
    #     #  cleaner.game_stats['successful_actions'] - int, total number of successful actions performed during active
    #     #                                                  turns
    #     #  cleaner.game_stats['recharge_count'] - int, number of turns spent at a charging station
    #     #  cleaner.game_stats['recharge_energy'] - int, total energy gained from the charging station
    #     #  cleaner.game_stats['visits'] - int, total number of squares visited (visiting the same square twice counts
    #     #                                      as one visit)

    #     # This fitness functions considers total number of cleaned squares.  This may NOT be the best fitness function.
    #     # You SHOULD consider augmenting it with information from other stats as well.  You DON'T HAVE TO make use
    #     # of every stat.

    #     fitness[n] = cleaner.game_stats['cleaned']

    return fitness


def newGeneration(old_population):

    # This function should return a tuple consisting of:
    # - a list of the new_population of cleaners that is of the same length as the old_population,
    # - the average fitness of the old population

    N = len(old_population)

    # Fetch the game parameters stored in each agent (we will need them to
    # create a new child agent)
    gridSize = old_population[0].gridSize
    nPercepts = old_population[0].nPercepts
    nActions = old_population[0].nActions
    maxTurns = old_population[0].maxTurns

    # At this point you should sort the old_population cleaners according to fitness, setting it up for parent
    # selection.

    fitness = evalFitness(old_population)

    # # THIS IS TO USE CUMULATIVE FITNESS AND TO NORMALISE TO 1
    # sum_fitness = sum(fitness)
    # normalized_fitness = [fit / sum_fitness for fit in fitness]
    # cumulative_distribution = np.cumsum(normalized_fitness)
    # print("CUMSUM RESULT::: ", cumulative_distribution)
    # fitness = cumulative_distribution

    sum_fitness = sum(fitness)
    new_fitness = []

    for value in fitness:
        new_fitness.append(value / sum_fitness)

    # NOT SURE IF THIS IS NECESSARY
    population_fitness = list(zip(fitness, old_population))
    population_fitness.sort(reverse=True, key=lambda x: x[0])
    # Extract the sorted population (chromosomes only) from the sorted list of tuples
    sorted_fitness = [individual[0] for individual in population_fitness]
    sorted_population = [individual[1] for individual in population_fitness]

    print("SORTED FITNESS", sorted_fitness)
    # print("SORTED POPULATION", sorted_population[0].chromosome)

    # Create new population list...
    new_population = list()
    for n in range(N):

        # Create a new cleaner
        new_cleaner = Cleaner(nPercepts, nActions, gridSize, maxTurns)

        # Here you should modify the new cleaner' chromosome by selecting two parents (based on their
        # fitness) and crossing their chromosome to overwrite new_cleaner.chromosome
        newParentsIndices = np.random.choice(
            len(old_population), size=2, replace=False, p=new_fitness)
        newParent1 = old_population[newParentsIndices[0]].chromosome
        newParent2 = old_population[newParentsIndices[1]].chromosome

        child = cross_over(newParent1, newParent2)
        mutatedChild = mutate(child)
        new_cleaner.chromosome = mutatedChild

        # Consider implementing elitism, mutation and various other
        # strategies for producing a new creature.

        # .
        # .
        # .

        # Add the new cleaner to the new population
        new_population.append(new_cleaner)

    # At the end you need to compute the average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)

    return (new_population, avg_fitness)


def cross_over(parent1, parent2):
    uniform_crossover = [random.randint(0, 1) for _ in range(len(parent1))]
    newChild = []

    for i in range(len(parent1)):
        if uniform_crossover[i] == 1:
            newChild.append(parent1[i])
        else:
            newChild.append(parent2[i])
    # print("new child: ", newChild)
    return newChild


def mutate(child):
    mutateLevel = 0.09
    random_decimal = round(random.uniform(0, 1), 2)
    if random_decimal < mutateLevel:
        k = random.randint(0, 63)
        v = np.random(0, 1)
        child[k] = v
    return child
