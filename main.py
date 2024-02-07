import sys
import random
import numpy as np

def find_neighbors_distance_1(vector):
    neighbors = []
    n = len(vector)
    for i in range(n):
        if vector[i] == 0:  # Проверка, чтобы изменить только позиции с 0 на 1
            modified_vector = vector.copy()
            modified_vector[i] = 1
            neighbors.append(modified_vector)
    return neighbors
def find_neighbors_distance_2(vector):
    neighbors = []
    n = len(vector)
    for i in range(n):
        for j in range(i + 1, n):
            neighbor = vector.copy()
            neighbor[i] = 1 - neighbor[i]  # Инвертирование i-го элемента
            neighbor[j] = 1 - neighbor[j]  # Инвертирование j-го элемента
            neighbors.append(neighbor)
    return neighbors
def find_neighbors_distance_3(vector):
    neighbors = []
    n = len(vector)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                neighbor = vector.copy()
                neighbor[i] = 1 - neighbor[i]  # Инвертирование i-го элемента
                neighbor[j] = 1 - neighbor[j]  # Инвертирование j-го элемента
                neighbor[k] = 1 - neighbor[k]  # Инвертирование k-го элемента
                neighbors.append(neighbor) #.tolist()
    return neighbors
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
def find_X(b, c, p, ro, vector):
    result = np.zeros((n, m))

    for j in range(len(b)):
        result[:, j] = b[j] - c[:, j] - p - ro
    max_value = np.max(result, axis=0)
    # Создаем матрицу x
    x = np.zeros(result.shape, dtype=int)
    for j in range(result.shape[1]):
        if max_value[j] > 0:
            max_indices = np.where(result[:, j] == max_value[j])[0]
            x[max_indices, j] = 1
    #x = np.multiply(x, vector.reshape(-1, 1))
    return x

def first_criterion(f , vector):
    new_b = find_new_b(b, c, vector)
    term1 = SP(new_b)
    # Вычисление второго слагаемого
    term2 = -np.sum(f * vector)
    # Вычисление итогового значения формулы
    result = term1 + term2
    return result

def VND():
    first_vector = generate_first_vector(n,facility_amount)
    first_new_b = find_new_b(b,c, first_vector)
    first_price = SP(first_new_b)
    best_price = first_price
    best_vector = first_vector
    for vector in find_neighbors_distance_1(first_vector):
        new_vector_b = find_new_b(b, c ,vector)
        temp_price = SP(new_vector_b)
        temp_ro = RF(new_vector_b)
        temp_X = find_X(b,c, temp_ro, temp_price, vector)




def generate_f(n):
    random.seed(42)
    return np.random.randint(0, 100, size=n)


# Данные для задачи
with open('input.txt', 'r') as file:
    line = file.readline().strip()
    numbers = line.split()
    # Присваиваем значения переменным
    n = int(numbers[0]) # кол-во предприятий
    m = int(numbers[1])  # кол-во клиентов
    facility_amount = int(numbers[2])  # кол-во открываемых предприятий

b = np.loadtxt('input_b.txt')
c = np.loadtxt('input_c.txt')

f = generate_f(n)
V = 100

#np.set_printoptions(threshold=sys.maxsize)


# first_vector = generate_first_vector(n,facility_amount)
# first_new_b = find_new_b(b,c, first_vector)
# best_price = SP(first_new_b)
# best_ro = RF(first_new_b)
# best_vector = first_vector
# print(best_price, best_vector, np.sum(best_vector), best_ro)
# for vector in find_neighbors_distance_1(first_vector):
#     new_vector_b = find_new_b(b, c, vector)
#     temp_price = SP(new_vector_b)
#     temp_ro = RF(new_vector_b)
#     temp_vector = vector
#     if temp_price > best_price:
#         best_price = temp_price
#         best_vector = temp_vector
#     best_ro = RF(new_vector_b)
# print(best_price, best_vector, np.sum(best_vector), best_ro)
# for vector in find_neighbors_distance_2(best_vector):
#
#     new_vector_b = find_new_b(b, c, vector)
#     temp_price = SP(new_vector_b)
#     temp_ro = RF(new_vector_b)
#     temp_vector = vector
#     if temp_price > best_price:
#         best_price = temp_price
#         best_vector = temp_vector
#     best_ro = RF(new_vector_b)
# print(best_price, best_vector, np.sum(best_vector), best_ro)
# for vector in find_neighbors_distance_3(best_vector):
#     new_vector_b = find_new_b(b, c, vector)
#     temp_price = SP(new_vector_b)
#     temp_ro = RF(new_vector_b)
#     temp_vector = vector
#     if temp_price > best_price:
#         best_price = temp_price
#         best_vector = temp_vector
#     best_ro = RF(new_vector_b)
#
# print(best_price, best_vector, np.sum(best_vector), best_ro)






