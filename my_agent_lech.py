__author__ = "Ben Knox"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "knobe957@student.otago.ac.nz"

import numpy as np

agentName = "<my_agent>"
# Train against random agent for 5 generations,
trainingSchedule = [("random_agent.py", 5), ("self", 1)]
# then against self for 1 generation

# This is the class for your cleaner/agent


class Cleaner:

    def __init__(self, nPercepts, nActions, gridSize, maxTurns):
        # Initialize the chromosome
        self.nPercepts = nPercepts
        self.nActions = nActions
        self.gridSize = gridSize
        self.maxTurns = maxTurns
        self.chromosome = self.createInitialChromosome()

    def createInitialChromosome(self):
        # Initialize chromosome with random weights and biases
        # Each action has (nPercepts + 1) weights (nPercepts for inputs and 1 for bias)
        chromosome = np.random.uniform(-1, 1,
                                       size=(self.nActions, self.nPercepts + 1))
        return chromosome

    def AgentFunction(self, percepts):
        # Extract percepts (visual, energy, bin, fails)
        visual, energy, bin, fails = percepts

        # Concatenate visual inputs with other percepts
        input_vector = np.concatenate(
            (visual.flatten(), [energy, bin, fails]))

        # Initialize empty action vector
        action_vector = np.zeros(self.nActions)

        # Calculate outputs for each action using the corresponding weights and biases
        for action in range(self.nActions):
            weights_and_bias = self.chromosome[action]
            output = np.dot(input_vector, weights_and_bias)
            action_vector[action] = output

        # Select the action with the highest output
        selected_action = np.argmax(action_vector)

        # Create a one-hot encoded action vector
        action_vector = np.zeros(self.nActions)
        action_vector[selected_action] = 1

        return action_vector


def evalFitness(population):

    N = len(population)

    # Fitness initialiser for all agents
    fitness = np.zeros((N))

    # This loop iterates over your agents in the old population - the purpose of this boilerplate
    # code is to demonstrate how to fetch information from the old_population in order
    # to score fitness of each agent
    for n, cleaner in enumerate(population):
        # cleaner is an instance of the Cleaner class that you implemented above, therefore you can access any attributes
        # (such as `self.chromosome').  Additionally, each object have 'game_stats' attribute provided by the
        # game engine, which is a dictionary with the following information on the performance of the cleaner in
        # the last game:
        #
        #  cleaner.game_stats['cleaned'] - int, total number of dirt loads picked up
        #  cleaner.game_stats['emptied'] - int, total number of dirt loads emptied at a charge station
        #  cleaner.game_stats['active_turns'] - int, total number of turns the bot was active (non-zero energy)
        #  cleaner.game_stats['successful_actions'] - int, total number of successful actions performed during active
        #                                                  turns
        #  cleaner.game_stats['recharge_count'] - int, number of turns spent at a charging station
        #  cleaner.game_stats['recharge_energy'] - int, total energy gained from the charging station
        #  cleaner.game_stats['visits'] - int, total number of squares visited (visiting the same square twice counts
        #                                      as one visit)

        # This fitness functions considers total number of cleaned squares.  This may NOT be the best fitness function.
        # You SHOULD consider augmenting it with information from other stats as well.  You DON'T HAVE TO make use
        # of every stat.

        fitness[n] = cleaner.game_stats['cleaned']

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

    fitness = evalFitness(old_population)

    # At this point you should sort the old_population cleaners according to fitness, setting it up for parent
    # selection.
    # .
    # .
    # .

    # Create new population list...
    new_population = list()
    for n in range(N):

        # Create a new cleaner
        new_cleaner = Cleaner(nPercepts, nActions, gridSize, maxTurns)

        # Here you should modify the new cleaner' chromosome by selecting two parents (based on their
        # fitness) and crossing their chromosome to overwrite new_cleaner.chromosome

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
