from main import *

def local_search_SP(y):
    best_y = y
    neighbors = k_shake(y, 1)
    for neighbor in neighbors:
        if SP(find_new_b(b, c, neighbor), neighbor) > SP(find_new_b(b, c, best_y), best_y):
            best_y = neighbor
    if np.array_equal(y, best_y):
        neighbors = k_shake(best_y, 2)
        for neighbor in neighbors:
            if SP(find_new_b(b, c, neighbor), neighbor) > SP(find_new_b(b, c, best_y), best_y):
                best_y = neighbor
    else:
        return best_y
    if np.array_equal(y, best_y):
        neighbors = k_shake(best_y, 3)
        for neighbor in neighbors:
            if SP(find_new_b(b, c, neighbor), neighbor) > SP(find_new_b(b, c, best_y), best_y):
                best_y = neighbor
    else:
        return best_y
    return best_y

def VND(Imax):
    I = 0
    count = 0
    y = generate_first_vector(n)
    while I < Imax and count < 2:
        previous_fitness = SP(find_new_b(b,c,y),y)
        y = local_search_SP(y)
        I += 1
        new_fitness = SP(find_new_b(b,c,y),y)
        if previous_fitness == new_fitness:
            count += 1
        else:
            count = 0
        print(new_fitness)
    return y
best_y = VND(500)
best_rho = RF(find_new_b(b,c,best_y), best_y)
print(best_rho)