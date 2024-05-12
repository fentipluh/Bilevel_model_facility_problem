
from mip import *
import numpy as np
import time
time_list = []
objective_list = []
for i in range(1,3):
    print(i)
    # Путь к файлу input.txt
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
    start_time = time.time()


    f = [0] * n
    V = 240
    difference = b - c

    # Находим максимальное значение по всем элементам полученной матрицы
    W = np.max(difference)

    model = Model(solver_name=GRB, sense= MAXIMIZE)
    model.verbose = 0
    #variables

    p = model.add_var()
    rho = model.add_var()
    x = [[model.add_var(var_type=BINARY) for j in range(m)] for i in range(n)]
    y = [ model.add_var(name = 'y', var_type=BINARY) for i in range(n) ]
    z = [[model.add_var() for j in range(m)] for i in range(n)]
    r = [[model.add_var() for j in range(m)] for i in range(n)]

    model.objective = maximize(rho) #1
    # constraints
    model += (xsum(z[i][j] for i in range (n) for j in range(m)) - xsum(f[i]*y[i] for i in range(n))) >= V #(2)


    for i in range(n):
        for j in range(m):
            model += x[i][j] <= y[i] #(4)

    for j in range(m):
        model += xsum(x[i][j] for i in range(n)) <= 1 #(5)

    for j in range(m):
        model += xsum((b[j]*x[i][j] - c[i][j]*x[i][j] - r[i][j] - z[i][j]) for i in range(n)) >= 0 #(6)

    for k in range(n):
        for j in range(m):
            model += xsum((c[i][j]*x[i][j]) for i in range(n)) <= c[k][j] + ((1-y[k])*W) #(7)

    for i in range(n):
        for j in range(m):

            model += ((1-x[i][j])*W + z[i][j]) >= p
            model += ((1-x[i][j])*W + p) >= z[i][j]
            model += z[i][j] <= (x[i][j] * W)
            model += z[i][j] >= 0

            model += ((1 - x[i][j]) * W + r[i][j]) >= rho
            model += ((1 - x[i][j]) * W + rho) >= r[i][j]
            model += r[i][j] <= (x[i][j] * W)
            model += r[i][j] >= 0

    model += rho >=0
    model += p >= 0



    #model.start = [1 for i in range(n)]
    model.preprocess = 1
    model.threads = -1

    model.optimize()
    objective_list.append(model.objective_value)
    print(model.objective_value)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ", round(execution_time,2), " seconds")
    time_list.append(round(execution_time,2))
for i in range(len(objective_list)):
    print(objective_list[i])
for i in range(len(time_list)):
    print(time_list[i])



