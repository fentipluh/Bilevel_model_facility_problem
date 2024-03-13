import numpy as np
from main import RF
from main import find_new_b

# Данные для задачи
with open('input.txt', 'r') as file:
    line = file.readline().strip()
    numbers = line.split()
    # Присваиваем значения переменным
    n = int(numbers[0])  # кол-во предприятий
    m = int(numbers[1])  # кол-во клиентов

b = np.loadtxt('input_b.txt')
c = np.loadtxt('input_c.txt')
#f = generate_f(n)
f = [0] * n
V = 5
population_size = 50
Imax = 10
def generate_first_population(population_size, n):
    first_population = np.random.randint(2, size=(population_size, n))
    return first_population
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def fitness_RF(b,c,population):
    fitness = []
    for i in range(len(population)):
        temp_b = find_new_b(b, c, population[i])
        fitness.append(RF(temp_b))
    return np.array(fitness)
def tournament_selection(population, fitness, tournament_size, num_parents):
    selected_parents = []
    for i in range(num_parents):
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament = population[tournament_indices]
        tournament_fitness = fitness[tournament_indices]
        winner = tournament[np.argmax(tournament_fitness)]
        selected_parents.append(winner)
    return np.array(selected_parents)


def tournament_selection(population, fitness, k):
    new_population = []
    population_size = len(population)

    while len(new_population) < population_size:
        tournament_indices = np.random.choice(population_size, k, replace=False)
        tournament = [population[i] for i in tournament_indices]

        tournament_fitness = [fitness[i] for i in
                              tournament_indices]  # Get the fitness values for the selected individuals
        best_index = np.argmax(tournament_fitness)  # Find the index of the best individual
        best_individual = tournament[best_index]  # Get the best individual from the tournament

        new_population.append(best_individual)  # Add the best individual to the new population

    return new_population


def crossover_population(population):
    new_population = []

    while len(new_population) < len(population):
        # Случайно выбираем двух индивидуумов для кроссовера
        indices = np.random.choice(range(len(population)), 2, replace=False)
        parent1, parent2 = population[indices[0]], population[indices[1]]

        # Применяем функцию кроссовера к выбранным родителям
        offspring1, offspring2 = crossover(parent1, parent2)

        # Добавляем потомство к новой популяции
        new_population.extend([offspring1, offspring2])

    # Обрезаем лишнее потомство, если его количество превысило исходную популяцию
    new_population = new_population[:len(population)]

    return new_population

def find_best(population):
    best_rho = 0
    best_y = []
    for i in range(len(population)):
        temp_b = find_new_b(b, c, population[i])
        if (best_rho < RF(temp_b)):
            best_rho = RF(temp_b)
            best_y = population[i]
    return best_rho, best_y




# first_population = generate_first_population(population_size,n)
# fitness_values_first = fitness_RF(b, c, first_population)
# print(first_population)
# print("######")
# temp_population = tournament_selection(first_population, fitness_values_first, 2)
# print(temp_population)
# print("######")
# print(crossover_population(temp_population))



def genetic(Imax, tournament_size):
    population = generate_first_population(population_size, n)
    for generation in range(Imax):
        fitness = fitness_RF(b, c, population)
        # Применяем турнирный отбор к текущей популяции
        selected_population = tournament_selection(population, fitness, tournament_size)
        # Применяем скрещивание к выбранной популяции
        population = crossover_population(selected_population)
    best_rho = find_best(population)[0]
    best_y = find_best(population)[1]
    return best_rho, best_y

print(genetic(10, 3))
