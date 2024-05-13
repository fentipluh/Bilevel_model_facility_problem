from main import *
import numpy as np
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
    f = [0] * n
    V = 240
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
    def local_search_RF(y):
        best_y = y
        neighbors = k_shake(y,1)
        for neighbor in neighbors:
            if RF(find_new_b(b,c,neighbor,m), neighbor, f, V,n ,m) > RF(find_new_b(b,c,best_y,m), best_y, f, V, n, m):
                best_y = neighbor
        if np.array_equal(y,best_y):
            neighbors = k_shake(best_y,2)
            for neighbor in neighbors:
                if RF(find_new_b(b, c, neighbor,m), neighbor, f, V, n, m) > RF(find_new_b(b, c, best_y,m), best_y, f, V, n, m):
                    best_y = neighbor
        else:
            return best_y
        if np.array_equal(y,best_y):
            neighbors = k_shake(best_y,3)
            for neighbor in neighbors:
                if RF(find_new_b(b, c, neighbor,m), neighbor, f, V, n, m) > RF(find_new_b(b, c, best_y,m), best_y, f, V, n, m):
                    best_y = neighbor
        else:
            return best_y
        return best_y


    def VND(Imax):
        I = 0
        count = 0
        y = generate_first_vector(n)
        while I < Imax and count < 1:
            previous_fitness = RF(find_new_b(b,c,y,m),y, f, V, n ,m)
            previous_y = y
            y = local_search_RF(y)
            I += 1
            new_fitness = RF(find_new_b(b,c,y,m),y,f,V,n,m)
            if hamming_distance(y,previous_y) == 0:
                count += 1
            else:
                count = 0
        return y
    best_y = VND(100)
    best_rho = RF(find_new_b(b,c,best_y, m), best_y, f, V, n, m)
    print(best_rho)
    objective_list.append(best_rho)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ", round(execution_time,2), " seconds")
    time_list.append(round(execution_time, 2))
for i in range(len(objective_list)):
    print(objective_list[i])
for i in range(len(time_list)):
    print(time_list[i])