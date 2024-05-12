from ortools.linear_solver import pywraplp
import numpy as np
import time
from ortools.sat.python import cp_model
start_time = time.time()

# Чтение данных из файлов
with open('input.txt', 'r') as file:
    data = file.readline().strip().split()
    n = int(data[0])
    m = int(data[1])

f = np.array([5] * n)
V = 22
c = np.loadtxt('input_c.txt')
b = np.loadtxt('input_b.txt')
b = b.astype(int)
c = c.astype(int)
difference = b - c
W = np.max(difference)
print("Максимальное значение (W) =", W)

# Создание решателя
solver = pywraplp.Solver.CreateSolver('SCIP')

# Определение переменных
x = [[solver.BoolVar(f'x[{i}][{j}]') for j in range(m)] for i in range(n)]
y = [solver.BoolVar(f'y[{i}]') for i in range(n)]
z = [[solver.NumVar(0, solver.infinity(), f'z[{i},{j}]') for j in range(m)] for i in range(n)]
r = [[solver.NumVar(0, solver.infinity(), f'z[{i},{j}]') for j in range(m)] for i in range(n)]
p = solver.NumVar(0, solver.infinity(), 'p')
rho = solver.NumVar(0, solver.infinity(), 'rho')
# Функция цели
solver.Maximize(rho)

# Ограничения
# Ограничение (2)
solver.Add(sum(z[i][j] for i in range(n) for j in range(m))
    - sum(f[i] * y[i] for i in range(n)) >= V)

# Ограничения (4) и (5)
for i in range(n):
    for j in range(m):
        solver.Add(x[i][j] <= y[i])

for j in range(m):
    solver.Add(sum(x[i][j] for i in range(n)) <= 1)

for j in range(m):
    solver.Add(sum(b[j]*x[i][j] - c[i][j]*x[i][j] - r[i][j] - z[i][j] for i in range(n)) >= 0)

for k in range(n):
    for j in range(m):
        solver.Add(sum(c[i][j]*x[i][j] for i in range(n)) <= c[k][j] + (1-y[k])*W)

for i in range(n):
    for j in range(m):
        solver.Add((1-x[i][j])*W + z[i][j] >= p)
        solver.Add((1-x[i][j])*W + p >= z[i][j])
        solver.Add(z[i][j] <= x[i][j] * W)
        solver.Add(z[i][j] >= 0)

        solver.Add((1 - x[i][j]) * W + r[i][j] >= rho)
        solver.Add((1 - x[i][j]) * W + rho >= r[i][j])
        solver.Add(r[i][j] <= x[i][j] * W)
        solver.Add(r[i][j] >= 0)
solver.Add(rho >= 0)
solver.Add(p >= 0)



status = solver.Solve()
print(solver.Objective().Value())
if status == pywraplp.Solver.OPTIMAL:
    print("Solution:")
    print("Objective value =", solver.Objective().Value())
    print("y values:")
    for i in range(n):
        print(f'y[{i}] = {y[i].solution_value()}')
else:
    print("The problem does not have an optimal solution.")

