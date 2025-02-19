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


def find_new_b(b, c, vector, m):
    if (sum(vector) != 0):
        strings = np.where(vector == 1)[0]
        new_c = c[strings, :]
        min_values = np.min(new_c, axis = 0)
        result = b - min_values
        result.sort()
        return result
    else:
        return [0]*m


def SP(new_b, y, f, n, m):
    total_cost = sum(f[i] * y[i] for i in range(len(f)))
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

def RF(new_b, y, f, V, n, m):
    total_cost = sum(f[i] * y[i] for i in range(len(f)))
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
