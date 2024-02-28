import sys
import random
import numpy as np

def hamming_distance(y1, y2):
    return np.count_nonzero(y1 != y2)
def hamming_weight(y):
    return np.count_nonzero(y)
def k_swap_neighborhood(y, r, k):
    neighborhoods = []
    n = len(y)
    for i in range(n):
        if y[i]:  # Facility is open
            for j in range(n):
                if not y[j]:  # Facility is closed
                    y_swap = np.copy(y)
                    y_swap[i] = False
                    y_swap[j] = True
                    if hamming_weight(y_swap) == r and hamming_distance(y, y_swap) == 2 * k:
                        neighborhoods.append(y_swap)
    return neighborhoods

def k_shake(vector, k):
    if k == 1:
        neighbors = []
        n = len(vector)
        for i in range(n):
            neighbor = vector.copy()
            neighbor[i] = 1 - neighbor[i]
            neighbors.append(neighbor)


    if k == 2:
        neighbors = []
        n = len(vector)
        for i in range(n):
            for j in range(i + 1, n):
                neighbor = vector.copy()
                neighbor[i] = 1 - neighbor[i]  # Инвертирование i-го элемента
                neighbor[j] = 1 - neighbor[j]  # Инвертирование j-го элемента
                neighbors.append(neighbor)

    if k == 3:
        neighbors = []
        n = len(vector)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    neighbor = vector.copy()
                    neighbor[i] = 1 - neighbor[i]  # Инвертирование i-го элемента
                    neighbor[j] = 1 - neighbor[j]  # Инвертирование j-го элемента
                    neighbor[k] = 1 - neighbor[k]  # Инвертирование k-го элемента
                    neighbors.append(neighbor)  # .tolist()
    return neighbors
def find_best_neighbor_SP(neighbors):
    best_vector = neighbors[0]
    for i in range(len(neighbors)):
        if (SP(neighbors[i]) > SP(best_vector)):
            best_vector = neighbors[i]
    return best_vector


def find_new_b(b, c, vector):
    strings = np.where(vector == 1)[0]
    new_c = c[strings, :]
    min_values = np.min(new_c, axis = 0)
    result = b - min_values
    result.sort()
    return result
def SP(new_b):
    m = len(new_b) - 1   # Размер вектора b
    i = 2
    p_star = new_b[0]
    count = 0
    max_income = p_star * m

    while i <= m:
        if i > m:
            break
        if new_b[i - 1] == new_b[i]:
            count += 1
        else:
            count = 0

        if new_b[i] * (m - i + 1 + count) > max_income:
            p_star = new_b[i]
            max_income = new_b[i] * (m - i + 1 + count)

        i += 1

    return max_income
def RF(new_b):
    i = 2
    rho_star = (new_b[0] - V / m)
    count = 0

    while i <= m:
        if i > m:
            break

        if new_b[i - 2] == new_b[i - 1]:
            count += 1
        else:
            count = 0

        if (new_b[i - 1] - V / (m - i + 1 + count)) > rho_star:
            rho_star = (new_b[i - 1] - V / (m - i + 1 + count))

        i += 1
    return rho_star
def generate_first_vector(n, facility_amount):
    vector = np.zeros(n)  # Создаем вектор размера n, заполненный нулями
    indices = np.random.choice(range(n), facility_amount, replace=False)  # Выбираем случайные индексы без повторений
    vector[indices] = 1  # Устанавливаем единички в выбранных индексах
    return vector


def first_criterion(f , vector):
    new_b = find_new_b(b, c, vector)
    term1 = SP(new_b)
    # Вычисление второго слагаемого
    term2 = -np.sum(f * vector)
    # Вычисление итогового значения формулы
    result = term1 + term2
    return result

def generate_f(n):
    random.seed(42)
    return np.random.randint(0, 100, size=n)
def local_search_RF(y):
    best_y = y
    neighbors = k_swap_neighborhood(y, 5, 1)
    best_b = find_new_b(b, c, best_y)
    for neighbor in neighbors:
        temp_b = find_new_b(b, c, neighbor)
        if(RF(temp_b) >= RF(best_b)):
            best_y = neighbor
            best_b = temp_b
    return best_y



# Данные для задачи
with open('input.txt', 'r') as file:
    line = file.readline().strip()
    numbers = line.split()
    # Присваиваем значения переменным
    n = int(numbers[0])  # кол-во предприятий
    m = int(numbers[1])  # кол-во клиентов
facility_amount = int(numbers[2])  # кол-во открываемых предприятий

b = np.loadtxt('input_b.txt')
c = np.loadtxt('input_c.txt')
T = 10
f = generate_f(n)
V = 100


def VND_2(Imax, k):
    I = 0
    y = generate_first_vector(n,5)  # Step 0: Generate initial Boolean vector y
    print(RF(find_new_b(b,c,y)))
    print(y)
    while I < Imax:
        y_star = local_search_RF(y)  # Step 1: Apply local search to 1-Swap(y)
        I += 1
        if I > Imax or hamming_distance(y, y_star) == 0:
            break  # If no improvement or maximum iterations reached, stop
    return y_star  # Return the best found facility location y

y_star = VND_2(100, 1)
print(y_star)
b_star = find_new_b(b,c,y_star)
print(RF(b_star))
