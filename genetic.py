import numpy as np
from main import RF
from main import find_new_b
from main import generate_first_vector

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
f = [15] * n
V = 30
population_size = 10
def generate_first_population(population_size, n):
    first_population = np.random.randint(2, size=(population_size, n))
    return first_population
def crossover(parent1, parent2):
    crossover_point_1 = np.random.randint(1, len(parent1))
    crossover_point_2 = np.random.randint(1, len(parent2))
    # print(f"crossover point 1: {crossover_point_1}")
    # print(f"crossover point 2: {crossover_point_2}")
    # print(f"parent1: {parent1}")
    # print(f"parent2: {parent2}")
    parent1_srez = parent1[crossover_point_1:crossover_point_2]
    parent2_srez = parent2[crossover_point_1:crossover_point_2]
    # print(f"parent1_srez: {parent1_srez}")
    # print(f"parent2_srez: {parent2_srez}")
    parent1[crossover_point_1:crossover_point_2] = parent2_srez
    parent2[crossover_point_1:crossover_point_2] = parent1_srez
    # print(f"parent1_new: {parent1}")
    # print(f"parent2_new: {parent2}")
    child1 = parent1
    child2 = parent2
    return child1, child2

def fitness_RF(b,c,population):
    fitness = []
    for i in range(len(population)):
        temp_b = find_new_b(b, c, population[i])
        fitness.append(RF(temp_b, population[i]))
    return np.array(fitness)


def tournament_selection(population, fitness, k):
    new_population = []
    population_size = len(population)
    while len(new_population) < population_size:
        tournament_indices = np.random.choice(population_size, k, replace=False)
        tournament = [population[i] for i in tournament_indices]

        tournament_fitness = [fitness[i] for i in tournament_indices]  # Get the fitness values for the selected individuals
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

def find_best(best_rho,best_y,population):
    for i in range(len(population)):
        temp_b = find_new_b(b, c, population[i])
        if (best_rho < RF(temp_b, population[i])):
            best_rho = RF(temp_b, population[i])
            best_y = population[i]
    return best_rho, best_y

def genetic(Imax, tournament_size):
    population = generate_first_population(population_size, n)
    best_rho = 0
    best_y = generate_first_vector(n)
    for generation in range(Imax):
        fitness = fitness_RF(b, c, population)
        # Применяем турнирный отбор к текущей популяции
        selected_population = tournament_selection(population, fitness, tournament_size)
        # Применяем скрещивание к выбранной популяции
        print(population)
        population = crossover_population(population)
        result = find_best(best_rho, best_y, population)
        print(f"result: {result}")
        best_rho = result[0]
        best_y = result[1]
    print(best_rho)
    print(best_y)
    return best_rho, best_y



genetic(10,5)
