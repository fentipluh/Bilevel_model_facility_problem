from main import *
import random
import matplotlib.pyplot as plt
time_list = []
objective_list = []
for i in range(1,3):
    print(i)
    # Путь к файлу input.txt
    file_path = f'C:/Users/Fentipluh/PycharmProjects/diploma/dataset/gen_20_20_{i}.txt'
    # Инициализация переменных
    n, m = 0, 0
    c = None
    b = None
    V = 240
    f = [0]*n

    with open(file_path, 'r') as file:
        # Чтение первой строки и извлечение значений n и m
        first_line = file.readline()
        n, m = map(int, first_line.split())
        # Чтение следующих n строк для создания матрицы A
        c = np.zeros((n, m), dtype=np.float64)  # Создание матрицы с нужным размером
        for i in range(n):
            line = file.readline()
            c[i] = np.array(line.split(), dtype=np.float64)

        # Чтение последней строки для получения массива b
        last_line = file.readline()
        b = np.array(last_line.split(), dtype=np.float64)

    start_time = time.time()
    def find_new_b_1(b, c, vector):
        if (sum(vector) != 0):
            strings = [index for index, value in enumerate(vector) if value == 1]
            new_c = c[strings, :]
            min_values = [min(row) for row in zip(*new_c)]
            result = b - min_values
            result.sort()
            return result
        else:
            return [0]*m


    # константы задачи
    ONE_MAX_LENGTH = n    # длина подлежащей оптимизации битовой строки

    # константы генетического алгоритма
    POPULATION_SIZE = 100   # количество индивидуумов в популяции
    P_CROSSOVER = 0.9      # вероятность скрещивания
    P_MUTATION = 0.1       # вероятность мутации индивидуума
    MAX_GENERATIONS = 100  # максимальное количество поколений

    # RANDOM_SEED = 42
    # random.seed(RANDOM_SEED)

    class FitnessMax():
        def __init__(self):
            self.values = [0]


    class Individual(list):
        def __init__(self, *args):
            super().__init__(*args)
            self.fitness = FitnessMax()


    def oneMaxFitness(individual):
        temp_b = find_new_b_1(b, c, individual)
        cost = SP(temp_b, individual, f, n, m)
        return (cost),

    def individualCreator():
        return Individual([random.randint(0, 1) for i in range(ONE_MAX_LENGTH)])

    def populationCreator(n = 0):
        return list([individualCreator() for i in range(n)])


    population = populationCreator(n=POPULATION_SIZE)
    generationCounter = 0

    fitnessValues = list(map(oneMaxFitness, population))

    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    maxFitnessValues = []
    meanFitnessValues = []

    def clone(value):
        ind = Individual(value[:])
        ind.fitness.values[0] = value.fitness.values[0]
        return ind

    def selTournament(population, p_len):
        offspring = []
        for n in range(p_len):
            i1 = i2 = i3 = 0
            while i1 == i2 or i1 == i3 or i2 == i3:
                i1, i2, i3 = random.randint(0, p_len-1), random.randint(0, p_len-1), random.randint(0, p_len-1)

            offspring.append(max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

        return offspring

    def cxOnePoint(child1, child2):
        s = random.randint(2, len(child1)-3)
        child1[s:], child2[s:] = child2[s:], child1[s:]

    # def cxOnePoint(child1, child2):
    #     # Create a list to store the crossover result
    #     # Create a list to store the crossover result
    #     result1 = []
    #     result2 = []
    #
    #     # Perform uniform crossover
    #     for gene1, gene2 in zip(child1, child2):
    #         if random.choice([True, False]):
    #             result1.append(gene2)
    #             result2.append(gene1)
    #         else:
    #             result1.append(gene1)
    #             result2.append(gene2)
    #
    #     return result1, result2


    def mutFlipBit(mutant, indpb=0.01):
        for indx in range(len(mutant)):
            if random.random() < indpb:
                mutant[indx] = 0 if mutant[indx] == 1 else 1


    fitnessValues = [individual.fitness.values for individual in population]
    best_b = [0]*m
    best_y = [0]*n
    count = 0
    while generationCounter < MAX_GENERATIONS and count < 1:
        generationCounter += 1
        offspring = selTournament(population, len(population))
        offspring = list(map(clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                cxOnePoint(child1, child2)

        for mutant in offspring:
            if random.random() < P_MUTATION:
                mutFlipBit(mutant, indpb=1.0/ONE_MAX_LENGTH)

        freshFitnessValues = list(map(oneMaxFitness, offspring))
        for individual, fitnessValue in zip(offspring, freshFitnessValues):
            individual.fitness.values = fitnessValue

        population[:] = offspring
        fitnessValues = [ind.fitness.values[0] for ind in population]
        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        #print(f"Поколение {generationCounter}: Макс приспособ. = {maxFitness}, Средняя приспособ.= {meanFitness}")
        best_index = fitnessValues.index(max(fitnessValues))
        #print("Лучший индивидуум = ", *population[best_index], "\n")
        string = [*population[best_index]]
        temp_b = find_new_b_1(b,c,string)
        if (hamming_distance(best_y, string)):
            count += 1
        else:
            count = 0
        if (SP(best_b, best_y, f,n,m) <= SP(temp_b, string,f,n,m)):
            best_b = temp_b
            best_y = string


    best_rho = RF(best_b, best_y, f, V, n, m)
    objective_list.append(best_rho)
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Поколение')
    plt.ylabel('Макс/средняя приспособленность')
    plt.title('Зависимость максимальной и средней приспособленности от поколения')
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ", execution_time, " seconds")
    time_list.append(round(execution_time, 2))
for i in range(len(objective_list)):
    print(objective_list[i])
for i in range(len(time_list)):
    print(time_list[i])