__author__ = "Ben Knox"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "knobe957@student.otago.ac.nz"

import numpy as np
import random

agentName = "<my_agent>"
# Train against random agent for 5 generations,
trainingSchedule = [("random_agent.py", 50), ("self", 50),
                    ("random_agent.py", 50)]


# trainingSchedule = [("random_agent.py", 0)]
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

        self.chromosome = self.createInitialChromosome()

    def createInitialChromosome(self):
        chromosome = np.empty(21)

        for i in range(21):
            chromosome[i] = np.random.uniform(-2, 2)

        return chromosome

    def AgentFunction(self, percepts):
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

        # 3x5 map where 1 indicates the location of energy station, 0 otherwise
        energy_locations = visual[:, :, 1]

        # 3x5 map of bots that can in this turn move up or down (from this bot's point of
        # view), -1 if the bot is an enemy, 1 if it is friendly
        vertical_bots = visual[:, :, 2]

        # 3x5 map of bots that can in this turn move up or down (from this bot's point
        # of view), -1 if the bot is an enemy, 1 if it is friendly
        horizontal_bots = visual[:, :, 3]

        # You may combine floor_state and energy_locations if you'd like: floor_state + energy_locations would give you
        floor_plus_energy = floor_state + energy_locations
        # a map where -1 indicates dirty square, 0 a clean one, and 1 an energy station.


        front_percep = floor_state[0:-1, 2]
        left_percep = floor_state[-1, :2]
        right_percep = floor_state[-1, -2:]

        back_percep = np.array(
            [vertical_bots[0, 2], horizontal_bots[1, 1], horizontal_bots[1, 3]])

        # You should implement a model here that translates from 'percepts' to 'actions'
        # through 'self.chromosome'.

        move_forward_array = np.concatenate(
            (front_percep.flatten() * self.chromosome[:2], [self.chromosome[2] * energy * bin]))

        turn_left_array = np.concatenate(
            (left_percep.flatten() * self.chromosome[5:7], [self.chromosome[7] * energy * bin]))

        turn_right_array = np.concatenate(
            (right_percep.flatten() * self.chromosome[8:10], [self.chromosome[10] * energy * bin]))

        move_back_array = np.concatenate(
            (back_percep * self.chromosome[11: 14], [self.chromosome[14] * energy * bin]))

        action_vector = np.zeros(4)

        action_vector = np.array([
            np.sum(move_forward_array) +
            np.sum(
                energy_locations[:-1, 1:4].flatten() * ((self.chromosome[15]) / energy) * ((self.chromosome[17]) / (bin + 1))),

            np.sum(turn_right_array) +
            np.sum(
                energy_locations[:, -2:].flatten() * ((self.chromosome[15]) / energy) * ((self.chromosome[17]) / (bin + 1))) +
            self.chromosome[18] * fails,

            np.sum(turn_left_array) +
            np.sum(
                energy_locations[:, 0:2].flatten() * ((self.chromosome[15]) / energy) * ((self.chromosome[17]) / (bin + 1))) +
            self.chromosome[19] * fails,

            np.sum(move_back_array) + self.chromosome[20] * (
                (self.chromosome[15]) / energy) * ((self.chromosome[17]) / (bin + 1))
        ])

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

        cleaned_squares = stats['cleaned']
        emptied_bins = stats['emptied']
        active_turns = stats['active_turns']
        successful_actions = stats['successful_actions']

        recharge_count = stats['recharge_count']
        recharge_energy = stats['recharge_energy']
        different_squares = stats['visits']

        weight_cleaned_squares = 19  
        weight_emptied_bins = 9
        weight_active_turns = 8
        weight_successful_actions = 8

        weight_recharge_count = 8
        weight_recharge_energy = 0
        weight_different_squares = 8  

        if different_squares < 4 or cleaned_squares == 0:
            fitness[n] = 1
        else:
            fitness[n] = (
                weight_cleaned_squares * cleaned_squares +
                weight_emptied_bins * emptied_bins +
                weight_active_turns * active_turns +
                weight_successful_actions * successful_actions +

                weight_recharge_count * recharge_count +
                weight_recharge_energy * recharge_energy +
                weight_different_squares * different_squares
            )

        if cleaned_squares > 5:
            fitness += 50
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

    num_elite = 4  # Number of top-performing individuals to preserve as elite
    elite_indices = np.argsort(fitness)[-num_elite:]

    sum_fitness = sum(fitness)
    new_fitness = []

    for value in fitness:
        new_fitness.append(value / sum_fitness)

    # Create new population list...
    new_population = list()
    for n in range(N):

        # Create a new cleaner
        new_cleaner = Cleaner(nPercepts, nActions, gridSize, maxTurns)

        # Elitism
        if n in elite_indices:
            new_cleaner.chromosome = old_population[n].chromosome
        else:
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


# Random point cross over
def cross_over(parent1, parent2):
    num_crossover_points = 1
    chromosome_length = len(parent1)
    crossover_points = sorted(random.sample(
        range(chromosome_length), num_crossover_points))
    newChild = []

    # Initialize variables for tracking which parent's genes to use
    current_parent = 1
    next_crossover_point = 0

    for i in range(chromosome_length):
        if next_crossover_point < len(crossover_points) and i == crossover_points[next_crossover_point]:
            # Switch to the other parent for the next segment
            current_parent = 3 - current_parent  # Toggle between 1 and 2
            next_crossover_point += 1

        # Append the gene from the current parent
        if current_parent == 1:
            newChild.append(parent1[i])
        else:
            newChild.append(parent2[i])

    return newChild


def mutate(child):
    mutateLevel = 0.07
    random_decimal = round(random.uniform(0, 1), 2)
    if random_decimal < mutateLevel:
        k = random.randint(0, 20)
        v = np.random.uniform(-1, 1)
        child[k] = v
    return child
