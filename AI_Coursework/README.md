**ga-continuous-distrib.py**:

A mostly complete example of a Genetic Algorithm optimised for continous problems. Implemented various crossover, selection and mutation methods:

Selection:
  - Roulette Style
  - Steady State Selection

Mutation:
  - Single gene mutation
  - Multi gene mutation

Crossover:
  - Middle point crossover
  - Multi point crossover

Various continuous optimisation problems were implemented:
  1. Sum square problem
  2. Weighted Input Problem
  3. Levy Function
  4. Trid Function
  5. Griewank Function
  6. Zakharov Function

A very bare bone test bench has been implemented, that allows for multiple GA runs, and their results to be displayed as a couple of graphs.


**ga-combinatorial-distrib.py + City.py**:

Another Genetic Algorithm that is for Combinatorial Problems, such as: 
  1. Sum Ones (Maximisation)
  2. Sum Ones (Minimisation)
  3. Reaching a Target String
  4. Reaching a Target Number in Hexadecimal (i.e. 255 --> 'FF')
  5. Knapsack Problem
  6. Travelling Salesman Problem

This particular implementation uses 'Tournament Selection' by default, to find the Parents for the next Generation. This also allows for 'Elitism' to be specified - In which the top ~10% of the population in terms of fitness, are saved for the next generation. This generally allows for a faster convergence to the ideal individual of a given problem.
 
For the Travelling Salesman Problem, due to the requirement that each individual must have every single gene only once, [Ordered Crossover](https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_crossover.htm) was implemented, as was a Gene-Swap Mutation (that simply swaps the position of 2 genes in the individual). The 'City' objects are used as genes for each indiviudal, for ease of implementation - Theoretically a Python dictionary, with the ID as the key, and a tuple representing Position as the value, could have been used, but that seems more complicated. After running, the final, best individual found from the GA, is visualised using Matplotlib.

Written in Python 3.8.7, uses Numpy 1.19.2 and Matplotlib 3.3.2
