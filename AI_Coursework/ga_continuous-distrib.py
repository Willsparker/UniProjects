# -*- coding: utf-8 -*-

# Coded in Python 3.8.7 64-Bit
# Use standard python package only.
import random 
import math
import numpy as np
import matplotlib.pyplot as plt
# This is purely so I can actually see the damn colours
import matplotlib as mpl
mpl.style.use('seaborn')

# This is to stop Division by Zero warnings that are already handled
import warnings
warnings.filterwarnings("ignore")

### Program Options ###

DEBUG = False
FITNESS_CHOICE = 1

#   FITNESS_CHOICE refers to which Continuous Optimization problem we want to solve
#   1. Sum squares
#   2. Weighted Numerical Inputs
#   3. Levy Function (http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2056.htm)
#   4. Trid Function (http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2904.htm)
#   5. Griewank Function (http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page1905.htm)
#   6. Zakharov Function (http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page3088.htm)

SELECTION_CHOICE = 1

#   SELECTION_CHOICE refers to how the population is manipulated in regards to the parents and offspring
#   1. Roulette Wheel Selection
#   2. Steady State Selection

MUTATION_CHOICE = 1

#   MUTATION_CHOICE refers to how the offspring is mutated.
#   1. Single gene mutation - One gene is picked at random, and changed by -5<= x <= 5
#   2. Multi gene mutation - Multiple genes are picked at random and changed by -5<= x <= 5

CROSSOVER_CHOICE = 1

#   CROSSOVER_CHOICE refers to how the offspring are produced from the parents
#   1. Middle point crossover - half of Parent 1 and half of Parent 2 are put together to create 2 new offspring
#   2. Multiple point crossover - 2 points are selected, and the genes will be from either parent 1 or parent 2.  

## Test Bed Options (Because I don't want to 'rigorously evaluate' my GA implementation manually)
RUN_TESTS = True
CHANGED_VAR = "CROSSOVER"
# Options include: POPULATION , MUTATION, CROSSOVER, PARENT_NO, NONE
# NOTE: "NONE" will just run NO_OF_TESTS runs of GA, not varying the variables
DIFFERENCE = 0.01
# Absolute difference between tests. i.e. if DIFFERENCE = 0.2 and CHANGED_VAR = "MUTATION",
# one run will have a mutation rate of 0.2, and the next would have one of 0.4
START_VAL = 0
# Starting value for CHANGED_VAR. i.e. if START_VAL = 50, DIFFERENCE = 10 and CHANGED_VAR = "POPULATION"
# The first run will have a population of 50, the next has 60, and the next has 70.
NO_OF_TESTS = 100
# The number of GA runs, with the varying var. i.e. NO_OF_TESTS = 3, START_VAL = 50, DIFFERENCE = 10 and CHANGED_VAR = "POPULATION"
# There'd be three runs of the GA, with the population at 50,60,70. If NO_OF_TESTS = 4, there'd be 4 at 50,60,70,80

## Default Global Variables 
POPULATION_SIZE = 30
GENERATIONS = 10
SOLUTION_FOUND = False

CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.4

# For the Weighted Input Problem
INPUT = None

# For the Test Bed
saved_population = None

## Additional Global Variables
PARENT_NO = 6
NO_OF_GENES = 6 
UPPERBOUND = 10
LOWERBOUND = -10

FITNESS_DICTIONARY = {
    1: "Sum Squares",
    2: "Weighted Input",
    3: "Levy Function",
    4: "Trid Function",
    5: "Griewank Function",
    6: "Zakharov Function"
}

# Generates a random list of numbers between -20 and 20, for FITNESS_CHOICE = 2
def generateRandomList():
    randList = []
    for _ in range(NO_OF_GENES):
        randList.append(random.randrange(-10,10))
    return randList

# Generates the population, given a set of boundaries
def generate_population(size, lower_bound, upper_bound):
    population = [generateIndividual(NO_OF_GENES, lower_bound, upper_bound) 
        for _ in range(size)]
    return population

# Generates an individual of the population
def generateIndividual(no_of_genes,lower_bound,upper_bound):
    # Debug code to ensure correct fitness implementation
    #return [ 0 for _ in range(no_of_genes)]
    #return [ 0,0,0,1,0,0]
    #return [ 6, 10, 12, 12, 10, 6]

    # Individual is a list of random ints
    return [random.randrange(lower_bound,upper_bound) for _ in range(no_of_genes)]
   
# Compute fitness - Depends on the implementation.
def compute_fitness(individual):
    switcher = {
        1: sum_square,
        2: weighted_input,
        3: levy,
        4: trid,
        5: griewank,
        6: zakharov
    }
    func = switcher.get(FITNESS_CHOICE)
    return func(individual)

def sum_square(individual):
    # We want the fitness to be larger, the better the implementation.
    # Therefore, the lesser values of 'fitness' are better here.
    fitness = sum([ (x+1)*individual[x]**2 for x in range(NO_OF_GENES)])
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness

def weighted_input(individual):
    # Maximization
    # No ideal solution
    return sum([individual[x]*INPUT[x] for x in range(NO_OF_GENES)])

def levy(individual):
    # Minimization problem
    # Ideal individual is all ones.
    weights = [ 1 + ((individual[i] - 1)/4) for i in range(NO_OF_GENES)]
    term_1 = (math.sin(math.pi*weights[0]))**2
    term_3 = (weights[-1]-1)**2 * (1+(math.sin(2*math.pi*weights[-1])**2))
    term_2 = sum([(weights[i]-1)**2 * (1+10*(math.sin(math.pi * weights[i]+1)**2)) for i in range(NO_OF_GENES-1)])
    
    try:
        fitness = abs(1/(term_1 + term_2 + term_3)) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness

def trid(individual):
    # ideal individual is dependent on the no. of genes
    # i.e. for n=6, ideal is [6,10,12,12,10,6] & ideal fitness = -50
    term_1 = sum([(individual[i]-1)**2 for i in range(NO_OF_GENES)])
    term_2 = sum([individual[i]*individual[i-1] for i in range(1,NO_OF_GENES)])
    fitness = term_1 - term_2

    ideal_fitness = -NO_OF_GENES*(NO_OF_GENES+4)*(NO_OF_GENES-1)/6

    try:
        fitness = abs(1/(fitness-ideal_fitness)) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness

def griewank(individual):
    term_1 = sum([(individual[i-1]**2)/4000 for i in range(1,NO_OF_GENES+1)])
    term_2 = math.prod([math.cos(individual[i-1]/math.sqrt(i)) for i in range(1,NO_OF_GENES+1)])
    
    fitness = term_1 - term_2 + 1
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness
    
def zakharov(individual):
    term_1 = sum([individual[i]**2 for i in range(NO_OF_GENES)])
    term_2 = (sum([0.5 * i * individual[i] for i in range(NO_OF_GENES)]))**2
    term_3 = (sum([0.5 * i * individual[i] for i in range(NO_OF_GENES)]))**4
    fitness = term_1 + term_2 + term_3
    try:
        fitness = abs(1/fitness) * 100
    except ZeroDivisionError:
        fitness = float('inf')
    return fitness

# Randomly select parents, weighted with their fitness
def selection(population, fitness, no_of_parents):
    parents = np.empty((no_of_parents,NO_OF_GENES))

    # Can't just overwrite 'fitness', as the 'del' command later on won't work otherwise.
    posFitness = fitness.copy()

    # If any fitness values are negative, find the smallest fitness value and add it to the rest of them.
    # The plus one is so there is not a 0 chance the lowest fitness is selected.
    if (sum(n < 0 for n in fitness) != 0):
        posFitness = [ fitness[x] + abs(min(fitness)) + 1 for x in range(len(fitness))]

    total_fitness = sum(posFitness)
    try:
        relative_fitness = [ (x/total_fitness) for x in posFitness ]
        indicies = np.random.choice(range(0,POPULATION_SIZE),size=no_of_parents,replace=False,p=relative_fitness)
    except ZeroDivisionError:
        print("Fitness of 0 in the population")
        indicies = np.random.choice(range(0,POPULATION_SIZE),size=no_of_parents,replace=False)
        return False

    parents = [population[x] for x in indicies]

    if SELECTION_CHOICE == 2:
        # Remove the weakest population members, as they will be replaced with the children
        idx = np.argpartition(fitness,no_of_parents)
        for i in sorted(idx[:no_of_parents],reverse=True):
            if DEBUG: print("Weakest POP: ", population[i], " Fitness: ", fitness[i])
            del population[i]
            del fitness[i]

    return parents
    
def crossover(parents, no_of_offspring):
    offspring = np.empty((no_of_offspring,NO_OF_GENES),dtype=int)

    for k in range(no_of_offspring):
        # Cyclic
        parent1_index = k%PARENT_NO
        parent2_index = (k+1)%PARENT_NO
        if CROSSOVER_CHOICE == 1:
            # single point crossover
            cross_point = int(NO_OF_GENES / 2)
            if random.random() < CROSSOVER_RATE:
                offspring[k] = list(parents[parent1_index][0:cross_point]) + list(parents[parent2_index][cross_point::])
            else:
                offspring[k] = parents[parent1_index]
        else:
            # 2-point crossover
            cross_points = sorted(random.sample(range(NO_OF_GENES),2))
            if random.random() < CROSSOVER_RATE:
                offspring[k] = list(parents[parent1_index][0:cross_points[0]]) + \
                    list(parents[parent2_index][cross_points[0]:cross_points[1]]) + \
                    list(parents[parent1_index][cross_points[1]::])
            else:
                offspring[k] = parents[parent1_index]
    return offspring

# Being 'creative', we're going to make it add a random number between -5 and 5
def mutation(offspring):
    no_of_mutations = 1 if MUTATION_CHOICE == 1 else 2
    for x in range(len(offspring)-1):
        if random.random() < MUTATION_RATE:
            random_index = random.sample(range(NO_OF_GENES),no_of_mutations)
            random_value = random.sample(range(-5,5), no_of_mutations)
            for i in range(len(random_index)):
                offspring[x][random_index[i]] += random_value[i]

    return offspring

def findBestInput(population,fitness):
    best_fitness = max(fitness)
    individual_index = fitness.index(best_fitness)
    return population[individual_index]

def next_generation(gen_count,fitness,parents,offspring):
    best_fitness = max(fitness)
    print("\nGeneration ", gen_count, ":",
            "\n\nSelected Parents: \n\n", parents,
            "\n\nOffspring (post crossover/mutation):\n\n", offspring,
            "\n\nFitness Values:\n\n", fitness,
            "\n\nBest Fitness after Generation ", gen_count, ": ", best_fitness)

# Returns true or false depending on if a perfect solution has been found
def check_solution(population):
    if FITNESS_CHOICE == 1 or FITNESS_CHOICE == 5 or FITNESS_CHOICE == 6:
        ideal_individual = [ 0 for x in range(NO_OF_GENES)]
    elif FITNESS_CHOICE == 2:
        # No ideal solution for this fitness choice
        return False
    elif FITNESS_CHOICE == 3:
        ideal_individual = [ 1 for x in range(NO_OF_GENES)]
    elif FITNESS_CHOICE == 4:
        ideal_individual = [ x*(NO_OF_GENES + 1 - x) for x in range(1,NO_OF_GENES+1)]
    
    for x in population:
        if len([i for i, j in zip(x, ideal_individual) if i == j]) == NO_OF_GENES:
            print("\nIdeal Individual Found!\n")
            return True
    
    return False

# Handles starting the genetic algorithm
def runGA(RUN_NO,**kwargs):
    lower_bound = LOWERBOUND
    upper_bound = UPPERBOUND
    global SOLUTION_FOUND
    global POPULATION_SIZE 
    global GENERATIONS
    global PARENT_NO
    global NO_OF_GENES
    global MUTATION_RATE
    global CROSSOVER_RATE
    global INPUT
    global saved_population
    
    # Optional arguments for Test Bed
    varToChange = kwargs.get('varToChange', None)
    value = kwargs.get('value', None)

    if varToChange == "POPULATION":
        POPULATION_SIZE = value
    elif varToChange == "MUTATION":
        MUTATION_RATE = value
    elif varToChange == "CROSSOVER":
        CROSSOVER_RATE = value
    elif varToChange == "PARENT_NO":
        PARENT_NO = value

    print("\nIndividual Run Settings:")
    print("POPULATION_SIZE: ", POPULATION_SIZE)
    print("PARENT_NUMBER: ", PARENT_NO)
    print("MUTATION_RATE: ", MUTATION_RATE)
    print("CROSSOVER_RATE: ", CROSSOVER_RATE)

    results_dict = {}
    gen_count = 1

    # This is to ensure that Test runs start with the same individual each time
    if RUN_NO == 0:
        INPUT = generateRandomList() # For Weighted Inputs Problem
        population = generate_population(POPULATION_SIZE, lower_bound, upper_bound)
        saved_population = population.copy()
    elif RUN_NO != 0 and (varToChange == "POPULATION"):
        population = generate_population(POPULATION_SIZE, lower_bound, upper_bound)
    else:
        population = saved_population.copy()
    
    fitness = [compute_fitness(x) for x in population]
    results_dict[gen_count] = max(fitness)
    if check_solution(population): 
        SOLUTION_FOUND = True
        print("Found after ", gen_count , " generation(s)")
    else:
        gen_count += 1 
    
    while (gen_count<=GENERATIONS and SOLUTION_FOUND != True): 
        ## Parent Selection Stage        
        parents = selection(population,fitness,PARENT_NO)
        if SELECTION_CHOICE == 1:
            # Roulette Wheel Selection --> NewPop = Parents + Offspring
            offspring = crossover(parents, POPULATION_SIZE - PARENT_NO)
            ## Mutation
            offspring = mutation(offspring)
            ## Survive
            population = list(parents) + list(offspring)
        else:
            # Steady State Selection --> NewPop = oldPop - weakest members + offspring
            offspring = crossover(parents, PARENT_NO)
            ## Mutation
            offspring = mutation(offspring)
            ## Survive
            population = list(population) + list(offspring)   
        
        fitness = [compute_fitness(x) for x in population]
        results_dict[gen_count] = max(fitness)

        if DEBUG: next_generation(gen_count,fitness,parents,offspring)

        if check_solution(population): 
            SOLUTION_FOUND = True
            print("Found after ", gen_count , " generation(s)")
        else:
            gen_count += 1 
        
        #offspring_fitness = [compute_fitness(x) for x in offspring]
        #population = list(population) + list(offspring)
        #fitness = offspring_fitness + fitness

    if FITNESS_CHOICE == 2: print("Input: ", INPUT)

    print("Best Individual: ", findBestInput(population,fitness))
    print("Best final outcome: ", max(fitness))  

    if SOLUTION_FOUND:
        # Remove the final entry's fitness, as it will be infinite (or disproportionately big)
        results_dict.pop(gen_count)
        SOLUTION_FOUND = False
    
    return results_dict

def main(): 
    global POPULATION_SIZE 
    global GENERATIONS
    global PARENT_NO
    global NO_OF_GENES
    global CHANGED_VAR

    if (PARENT_NO > POPULATION_SIZE) or (RUN_TESTS == True and CHANGED_VAR == "POPULATION" and START_VAL <= PARENT_NO):
        print("ERROR: PARENT_NO must be less than POPULATION_SIZE")
        return
    if (PARENT_NO%2 != 0) or (CHANGED_VAR == "POPULATION" and START_VAL%2 !=0):
        # We're assuming throuples don't exist in this world
        print("WARNING: PARENT_NO must be even\n")
        PARENT_NO +=1
    
    if NO_OF_GENES%2 != 0:
        # This is due to the crossover method used
        print("WARNING: NO_OF_GENES must be even\n")
        NO_OF_GENES +=1

    print("Global Run Settings: ")
    print("DEBUG: ", DEBUG)
    print("FITNESS_CHOICE: ", FITNESS_DICTIONARY[FITNESS_CHOICE])
    print("GENERATION_NUMBER: ", GENERATIONS)
    print("NUMBER OF GENES: ", NO_OF_GENES)
    print("INITIALISATION BOUNDARIES: ", LOWERBOUND, " To ", UPPERBOUND)
    
    if RUN_TESTS:
        results_dict = [dict() for _ in range(NO_OF_TESTS)]
        for x in range(NO_OF_TESTS):
            value = x*DIFFERENCE+START_VAL
            print("\nTEST RUN: ", x+1, "\n")
            results_dict[x] = runGA(x,varToChange=CHANGED_VAR,value=value)
    else:
        results_dict = [dict()]
        results_dict[0] = runGA(0)
    
    ###
    # The Stupid Graph Section
    ###
    # This isn't a great implementation, but hey, it will do :shrug

    # Prints all the runs, on a fitness against number of generations graph
    max_generations = range(1,GENERATIONS+1)
    plt.figure(1)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    for x in range(len(results_dict)):
        values = results_dict[x].values()
        if RUN_TESTS:
            # Weird string format thing to get round lack of double precision
            line_label = CHANGED_VAR + ": " + "{:.2f}".format(x*DIFFERENCE+START_VAL)
            plt.plot(max_generations[0:len(results_dict[x].keys())],values, label=line_label)
            # If the run got an ideal individual (as we remove that point)
            if len(results_dict[x].keys()) < GENERATIONS:
                plt.plot(max(results_dict[x].keys()),max(values), marker='o', markersize=5, color="black")
            if CHANGED_VAR != "NONE" and NO_OF_TESTS <= 10: plt.legend() 
        else:
            plt.plot(max_generations[0:len(results_dict[x].keys())],values)
    
    if RUN_TESTS:
        test_range= range(NO_OF_TESTS)
        x_axis = [ x*DIFFERENCE+START_VAL for x in test_range ]
        if CHANGED_VAR == "NONE":
            x_axis = test_range
            CHANGED_VAR = "Run Number"
        # When points are infinite, its looks better to have the line at the
        # highest point in the graph and indicate infinite with a black dot.
        max_y = max([ max(results_dict[x].values()) for x in test_range ])

        # If the size of the results dict is less, we know that that run got
        # to infinite, therefore note their x coord
        inf_points = [ x*DIFFERENCE+START_VAL for x in test_range if len(results_dict[x].keys()) < GENERATIONS]

        # Print the points for the results - if inf, plot it's y value as max_y
        fig2_y_axis = [ max(results_dict[x].values()) if x*DIFFERENCE+START_VAL not in inf_points else max_y for x in test_range ]

        plt.figure(2)
        plt.xlabel(CHANGED_VAR)
        plt.ylabel("Fitness")
        plt.plot(x_axis,fig2_y_axis, color="green")
        plt.plot(x_axis,fig2_y_axis, marker='o', markersize=5, color="green")
        # Plot the inf markers
        for x in inf_points:
            plt.plot(x,max_y, marker='o', markersize=5, color="black")
    
    plt.show()

if __name__ == '__main__': 
    main() 
    
# Helpful Resources Used:
#   https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6 --> gave me the main structure
#   https://ai.stackexchange.com/a/3429 --> Suggestions for mutations
#   https://www.mdpi.com/2078-2489/10/12/390/pdf --> Basis for how crossover should work
