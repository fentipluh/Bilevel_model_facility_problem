import sys
import random
import numpy as np
import time

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
    if (sum(vector) != 0):
        strings = np.where(vector == 1)[0]
        new_c = c[strings, :]
        min_values = np.min(new_c, axis = 0)
        result = b - min_values
        result.sort()
        return result
    else:
        return [0]*m


def SP(new_b, y):
    total_cost = sum(f[i] * y[i] for i in range(n))
    i = 2
    p_star = new_b[0]
    count = 0
    max_income = p_star * m
    while i <= m:
        if i > m:
            break
        if new_b[i - 2] == new_b[i - 1]:
            count += 1
        else:
            count = 0
        if  new_b[i - 1] * (m - i + 1 + count) > max_income:
            p_star = new_b[i-1]
            max_income = new_b[i - 1] * (m - i + 1 + count)
        i += 1
    return p_star - total_cost

def RF(new_b, y):
    total_cost = sum(f[i] * y[i] for i in range(n))
    i = 2
    rho_star = (new_b[0] - (V + total_cost)/ m)
    count = 0
    while i <= m:
        if i > m:
            break

        if new_b[i - 2] == new_b[i - 1]:
            count += 1
        else:
            count = 0

        if (new_b[i - 1] - (V+total_cost) / (m - i + 1 + count)) > rho_star:
            rho_star = (new_b[i - 1] - (V + total_cost) / (m - i + 1 + count))

        i += 1
    return rho_star
def generate_first_vector(n):
    return np.random.randint(0, 2, size=n)

def generate_f(n):
    random.seed(42)
    return np.random.randint(0, 100, size=n)

# Путь к файлу input.txt
for i in range(1,3):
    file_path = f'C:/Users/Fentipluh/PycharmProjects/diploma/dataset/gen_100_100_{i}.txt'

# Инициализация переменных
n, m = 0, 0
c = None
b = None

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



f = [0] * n
V = 240
total_cost = 0

def VND_1(Imax, k):
    I = 0
    y = generate_first_vector(n)
    while I < Imax:
        y_star = local_search_SP(y, k)
        I += 1
        if I > Imax or hamming_distance(y, y_star) == 0:
            break
    return y_star

def VND_2(Imax, k):
    I = 0
    y = generate_first_vector(n)
    while I < Imax:
        y_star = local_search_RF(y, k)
        I += 1
        if I > Imax or hamming_distance(y, y_star) == 0:
            break

    return y_star
def main():
     y_star = VND_2(1000, 3)
     print(y_star)
     b_star = find_new_b(b,c,y_star)
     print(RF(b_star, y_star))
#main()