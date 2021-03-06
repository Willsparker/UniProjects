# -*- coding: utf-8 -*-

# Coded in Python 3.8.7 64-Bit
import random 
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')

POPULATION_SIZE = 10
GENERATIONS = 300
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.6
NO_OF_GENES = 10    # This var is overwritten for FITNESS_CHOICE 3 & 4
NO_OF_PARENTS = 3
DEBUG = False

# Elitism - Keep the best 10 percent of any population to go to the next generation
# Fancied coding this as I hadn't done this in Task 1.
ELITISM = True

# The size of the random population sample when selecting parents.
# Must be less than or equal to (Population size - PARENT_NO) 
TOURNAMENT_SELECTION_NUMBER = 6

# The Combinatorial Optimisation problem the GA needs to solve.
FITNESS_CHOICE = 6
# Choice list:
    # 1. Sum ones in a binary string - Minimization (constrained set being 0,1 (Base 2))
    # 2. Sum ones in a binary string - Maximization (constrained set being 0,1 (Base 2))
    # 3. Try to reach an sentence                   (constrained set being the lowercase alphabet + some grammar)
    # 4. Reaching a target number in Hex            (constrained set being Base 16)
    # 5. Knapsack problem                           (constrained by the randomly generated 'items')
    # 6. Travelling Salesman (task 3a)              (constrained by the randomly generated 'cities')

# Used for nice output
FITNESS_DICTIONARY = {
    1: "Sum Ones (Minimisation)",
    2: "Sum Ones (Maximisation)",
    3: "Reaching a Target String",
    4: "Reaching a Target Number (in Hexadecimal)",
    5: "Knapsack Problem",
    6: "Travelling Salesman"
}

# This is used to represent the different discreet domains, dependent on FITNESS_CHOICE
GENES = ''

# For FITNESS_CHOICE = 3
TARGET_SENTENCE = "Lorem ipsum dolor sit amet" # This needs to be larger than a single

# For FITNESS_CHOICE = 4
TARGET_NUMBER = 1286058684673 # This needs to be above 16 (as NO_OF_GENES must be >1)

# For FITNESS_CHOICE = 5
KNAPSACK_ITEMS = {}         # Random every run. Key : <ITEM_ID>, Value: (<WEIGHT>,<VALUE>)
KNAPSACK_THRESHOLD = 20     # Make weight the knapsack can take

# For FITNESS_CHOICE = 6 (i.e. Travelling Salesman)
import City
BOARD_SIZE = 100            # The max x and y of where the cities can be
NO_OF_CITIES = NO_OF_GENES  # Reads better in the code

def generate_population(size):
    # For Populations that are constrained by binary
    population = [ [ random.randint(0,1) for _ in range(NO_OF_GENES)] for _ in range(POPULATION_SIZE)]
    if FITNESS_CHOICE == 3:
        # For the population constrained by the characters in the "GENES" variable
        population = [ [ random.choice(GENES) for _ in range(NO_OF_GENES)] for _ in range(POPULATION_SIZE)]
    if FITNESS_CHOICE == 6:
        # For the travelling salesman problem - create NO_OF_CITIES Cities, at random points.
        population = [ [ City.City(x,random.randrange(0,BOARD_SIZE),random.randrange(0,BOARD_SIZE)) for x in range(NO_OF_CITIES)] for _ in range(POPULATION_SIZE) ]
    return population

def compute_fitness(individual):
    # A cleaner way of separating the different problems the GA can solve
    switcher = {
        1: sum_ones,
        2: sum_ones,
        3: match_sentence,
        4: match_number,
        5: knapsack,
        6: travelling_salesman
    }
    func = switcher.get(FITNESS_CHOICE)
    return func(individual)

def sum_ones(individual):
    # Steps through individual and if it's equal to the digit it's supposed to be, add 1 to fitness
    fitness = 0
    # Reduce code base by having both the minimization and maximization problem use this function
    digit = 0 if FITNESS_CHOICE == 1 else 1
    for x in individual:
        if x == digit:
            fitness += 1
    return fitness

def match_sentence(individual):
    # Add one to fitness if the character in the individual is the same, and in same position, as target sentence
    fitness = 0
    for index in range(len(individual)):
        if individual[index] == TARGET_SENTENCE[index]:
            fitness += 1

    return fitness

def match_number(individual):
    fitness = 0
    for x in range(len(individual)):
        if individual[x] == TARGET_NUMBER[x]:
            fitness += 1

    return fitness 

def knapsack(individual):
    sackValue = 0
    sackWeight = 0
    # Find total value and total weight of knapsack
    for x in range(len(individual)):
        if individual[x] == 1:
            sackWeight += KNAPSACK_ITEMS[x][0]
            sackValue += KNAPSACK_ITEMS[x][1]
    
    # If the weight is above the threshold, this isn't a viable solution
    if sackWeight > KNAPSACK_THRESHOLD:
        return 0
    else:
        return sackValue

def travelling_salesman(individual):
    fitness = 0
    # For each of the cities
    for x in range(len(individual)):
        try:
            city1 = individual[x]
            city2 = individual[x+1]
        except IndexError:
            city2 = individual[0]
        
        # Add to the fitness, the cost from city 1 to city 2
        fitness += city1.cost_to_city(city2)
    
    # A higher fitness is better - therefore we need to inverse this
    # (The '* 10000' is to make the numbers more meaningful)
    fitness = abs(1/fitness) * 10000
    return fitness

# Tournament Selection, as I did Roulette Wheel selection in Task 1.
def selection(population, fitness):
    parents = []

    # For each parent
    for _ in range(NO_OF_PARENTS):
        # Get TOURNAMENT_SELECTION_NUMBER random indexes of the population and their fitness
        indicies = random.sample(range(0,len(population)),TOURNAMENT_SELECTION_NUMBER)
        random_fitness = [ fitness[x] for x in indicies]
        
        # Add to parents, the indiviudal with the highest fitness of the randomly selected group
        parents.append(population[indicies[random_fitness.index(max(random_fitness))]])
        # And delete from the population
        del population[indicies[random_fitness.index(max(random_fitness))]]

    return parents

# Crossover selection depends on the problem the GA needs to solve.
def crossover(parents, no_of_offspring):
    offspring = []
    for k in range(no_of_offspring):
        # Cyclic <-- That's a good word
        p1_indx = k%NO_OF_PARENTS
        p2_indx = (k+1)%NO_OF_PARENTS
        if FITNESS_CHOICE == 6: 
            # Need ordered crossover for Travelling Salesman.
            if random.random() < CROSSOVER_RATE:
                indiv_offspring = [0 for _ in range(NO_OF_CITIES)] # Initialise it so I can change variables at specific places

                # Get 2 points, crspt_1 is always going to be less than crspt_2
                crspt_1 = random.randint(0,NO_OF_GENES-2)
                crspt_2 = 0
                while crspt_2 <= crspt_1:
                    crspt_2 = random.randint(1,NO_OF_GENES-1)

                # Set the new offspring to have the cities between the crosspoints of parent 1
                indiv_offspring[crspt_1:crspt_2] = parents[p1_indx][crspt_1:crspt_2]
            
                # (Just trust me, this works)
                # Start at Parent 2's 2nd cross point, add city if it's ID doesn't already appear in the new offspring
                off_count = 0
                par_count = 0
                # Repeat until the new offspring has the required amount of cities.
                while len([x for x in indiv_offspring if type(x) == City.City]) != NO_OF_CITIES:
                    # Next position of parent 2 to check
                    parent_index = (crspt_2+par_count)%NO_OF_CITIES
                    city_ids = [ x.id for x in indiv_offspring if type(x) == City.City]
                    # If parent 2's city ID at index 'parent_index' is not already in the new offspring
                    if not parents[p2_indx][parent_index].id in city_ids:
                        # Add the City in parent 2's parent_index, to the next available space in the new offspring
                        offspring_index = (crspt_2+off_count)%NO_OF_CITIES
                        indiv_offspring[offspring_index] = parents[p2_indx][parent_index]
                        off_count += 1
                
                    par_count += 1

                # Useful Debug for confirming the crossover selection works.

                #print(crspt_1)
                #print(crspt_2)
                #print([x.id for x in parents[p1_indx]])
                #print([x.id for x in parents[p2_indx]])
                #print([x.id for x in indiv_offspring])
            else:
                # The new offspring is the same as the parent, if the crossover rate comparison fails
                indiv_offspring = parents[p1_indx]
            
            offspring.append(indiv_offspring)
        else:
            # For non travelling-salesman problems, simple single-point crossover
            cross_point = random.randint(1,NO_OF_GENES-1)
            if random.random() < CROSSOVER_RATE:
                offspring.append(list(parents[p1_indx][0:cross_point]) + list(parents[p2_indx][cross_point::]))
            else:
                offspring.append(parents[p1_indx])

    return offspring

# Various mutation methods, depending on the fitness choice.
def mutation(individual):
    if random.random() < MUTATION_RATE:
        affected_gene = random.randint(0,NO_OF_GENES-1)
        if FITNESS_CHOICE == 3 or FITNESS_CHOICE == 4:
            # Set the affected gene to a randomly selected character
            individual[affected_gene] = random.choice(GENES)
        elif FITNESS_CHOICE == 6:
            # Swap mutation - select another random gene and swap the two - required for Travelling salesman.
            second_affected_gene = random.randint(0,NO_OF_GENES-1)
            while second_affected_gene == affected_gene:
                second_affected_gene = random.randint(0,NO_OF_GENES-1)
            
            temp = individual[affected_gene]
            individual[affected_gene] = individual[second_affected_gene]
            individual[second_affected_gene] = temp
        else:
            # Bit-flip mutation for the problems where the set constraint is binary digits
            individual[affected_gene] = 0 if individual[affected_gene] == 1 else 1
            
    return individual

def check_solution(population):
    ideal_individual = []
    if FITNESS_CHOICE == 1:
        ideal_individual = [0 for _ in range(NO_OF_GENES)]
    elif FITNESS_CHOICE == 2:
        ideal_individual = [1 for _ in range(NO_OF_GENES)]
    elif FITNESS_CHOICE == 3:
        ideal_individual = TARGET_SENTENCE
    elif FITNESS_CHOICE == 4:
        ideal_individual = list(TARGET_NUMBER)
    elif FITNESS_CHOICE == 5 or FITNESS_CHOICE == 6:
        # No algorithmic way of finding the ideal individual, especially when individuals are randomised
        return False

    for x in population:
        if x == ideal_individual:
            return True
    
    return False

SOLUTION_FOUND = False

def check_vars():
    global NO_OF_GENES
    global TARGET_SENTENCE
    global TARGET_NUMBER
    global NO_OF_PARENTS
    global POPULATION_SIZE
    global GENES

    if POPULATION_SIZE <= NO_OF_PARENTS:
        print("NO_OF_PARENTS must be less than the POPULATION_SIZE")
        return False
    
    if TOURNAMENT_SELECTION_NUMBER >= POPULATION_SIZE-NO_OF_PARENTS:
        print("The TOURNAMENT_SELECTION_NUMBER must be more than POPULATION_SIZE - NO_OF_PARENTS")
        return False

    if FITNESS_CHOICE == 3:
        print("Target Sentence: ", TARGET_SENTENCE)
        TARGET_SENTENCE = list(TARGET_SENTENCE.lower())
        NO_OF_GENES = len(TARGET_SENTENCE)
        GENES = '''abcdefghijklmnopqrstuvwxyz '_[]()?!<>.,'''
    
    if FITNESS_CHOICE == 4:
        print("Target Number in Denary: ", TARGET_NUMBER)
        # The '[2::]' removes the '0x' from the beginning if the hex() output
        TARGET_NUMBER = hex(TARGET_NUMBER)[2::]
        print("Target Number in Hex: ", TARGET_NUMBER)
        NO_OF_GENES = len(TARGET_NUMBER)
        GENES = '''1234567890abcdef'''

    if FITNESS_CHOICE == 5:
        for x in range(NO_OF_GENES):
            # Create NO_OF_GENES random items, and assign them random weights and values
            KNAPSACK_ITEMS[x] = (random.randrange(1,10),random.randrange(0,500))

    if NO_OF_GENES <= 1:
        print("NO_OF_GENES must be <1")
        if FITNESS_CHOICE == 3:
            print("Please input a larger string")
        if FITNESS_CHOICE == 4:
            print("Please input a larger number")
        return False

    return True


def main(): 
    global POPULATION_SIZE 
    global GENERATIONS
    global SOLUTION_FOUND
    global FITNESS_CHOICE

    # Check for valid vars
    if not check_vars(): exit(127)

    # Debug Output
    print("Running with following Variables:", \
            "\nFITNESS_CHOICE: ", FITNESS_DICTIONARY[FITNESS_CHOICE], \
            "\nELITISM: ", ELITISM, \
            "\nDEBUG: ", DEBUG, \
            "\nGENERATIONS: ", GENERATIONS, \
            "\nNO_OF_GENES: ", NO_OF_GENES, \
            "\nNO_OF_PARENTS: ", NO_OF_PARENTS, \
            "\nMUTATION_RATE: ", MUTATION_RATE, \
            "\nCROSSOVER_RATE: ", CROSSOVER_RATE, \
            "\n")

    if FITNESS_CHOICE == 5:
        print("Randomly Generated Knapsack Items: ")
        for x in KNAPSACK_ITEMS:
            print("ID: ", x, "  Weight: ", KNAPSACK_ITEMS[x][0], "  Value: ", KNAPSACK_ITEMS[x][1])
        print("")
    
    # Initial Population
    gen_count = 0
    population = generate_population(POPULATION_SIZE)
    # Compute intitial pop fitness
    fitness = [compute_fitness(x) for x in population]
    # Check solution 
    if DEBUG:
        print("POPULATION: ", population)
        print("FITNESS: ", fitness, "\n")

    if check_solution(population):
        print("Ideal Individual found in ", gen_count, " generations")
        SOLUTION_FOUND = True
    else:
        gen_count += 1

    while (gen_count <= GENERATIONS and SOLUTION_FOUND != True):
        next_generation = []
        
        if ELITISM:
            N = int((10*POPULATION_SIZE)/100)
            # Get the top N population, by fitness ( This helped: https://www.geeksforgeeks.org/python-indices-of-n-largest-elements-in-list/ )
            res = sorted(range(len(fitness)), key = lambda sub: fitness[sub])[-N:]
            next_generation += [ population[x] for x in res ]

        parents = selection(population,fitness)
        # If Elitism, we need more offspring then if Not Elitism
        offspring = crossover(parents,POPULATION_SIZE-len(next_generation))
        offspring = [ mutation(x) for x in offspring ]

        next_generation += offspring
        population = next_generation
        
        fitness = [compute_fitness(x) for x in population]
        fitness_index = fitness.index(max(fitness))
        best_individual = population[fitness_index] if FITNESS_CHOICE != 3 else ''.join(population[fitness_index])
        if FITNESS_CHOICE == 6:
            best_individual = [ "ID: " + str(x.id) for x in best_individual ]
        print("Generation: ", gen_count, "  Max Fitness: ", max(fitness), " Best Individual: ", best_individual)
        if DEBUG:
            print("POPULATION: ", population)
            print("FITNESS: ", fitness, "\n")
        if check_solution(population):
            print("Ideal Individual found in ", gen_count, " generations")
            SOLUTION_FOUND = True
        else:
            gen_count += 1
    
    # Visualise the Travelling Salesman Problem 
    if FITNESS_CHOICE == 6:
        # Plot lines for each 2 coords
        for x in range(len(population[fitness_index])):
            pt1 = population[fitness_index][x]
            try:
                pt2 = population[fitness_index][x+1]
            except IndexError:
                pt2 = population[fitness_index][0]
            
            plt.plot([pt1.pos[0],pt2.pos[0]],[pt1.pos[1],pt2.pos[1]])

        # Plot individual points on the 'board'
        points = [ x.pos for x in population[fitness_index] ]
        x,y = zip(*points)
        plt.scatter(x,y,s=40)

        for x in population[fitness_index]:
            # Annotate the City IDs
            plt.annotate(x.id, x.pos)

        plt.show() 
 
if __name__ == '__main__': 
    main() 