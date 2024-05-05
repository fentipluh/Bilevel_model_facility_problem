from main import *
import numpy as np

def local_search_RF(y):
    best_y = y
    neighbors = k_shake(y,1)
    for neighbor in neighbors:
        if RF(find_new_b(b,c,neighbor), neighbor) > RF(find_new_b(b,c,best_y), best_y):
            best_y = neighbor
    if np.array_equal(y,best_y):
        neighbors = k_shake(best_y,2)
        for neighbor in neighbors:
            if RF(find_new_b(b, c, neighbor), neighbor) > RF(find_new_b(b, c, best_y), best_y):
                best_y = neighbor
    else:
        return best_y
    if np.array_equal(y,best_y):
        neighbors = k_shake(best_y,3)
        for neighbor in neighbors:
            if RF(find_new_b(b, c, neighbor), neighbor) > RF(find_new_b(b, c, best_y), best_y):
                best_y = neighbor
    else:
        return best_y
    return best_y


def VND(Imax):
    I = 0
    count = 0
    y = generate_first_vector(n)
    while I < Imax and count < 2:
        previous_fitness = RF(find_new_b(b,c,y),y)
        y = local_search_RF(y)
        I += 1
        new_fitness = RF(find_new_b(b,c,y),y)
        if previous_fitness == new_fitness:
            count += 1
        else:
            count = 0
    return y
best_y = VND(100)
best_rho = RF(find_new_b(b,c,best_y), best_y)
print(best_rho)