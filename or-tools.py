from itertools import product
from sys import stdout as out

from mip import *
import numpy as np

with open('input.txt', 'r') as file:
    data = file.readline().strip().split()
    n = int(data[0])
    m = int(data[1])

f= [10] * n

V = 10

c = np.loadtxt('input_c.txt')
b = np.loadtxt('input_b.txt')
W = 10000

model = Model(solver_name="CBC")
model.verbose = 0
x = [[model.add_var(var_type=BINARY) for j in range(m)] for i in range(n)]
p = model.add_var(var_type=INTEGER)
y = [model.add_var(var_type=BINARY) for i in range(n)]
rho = model.add_var()
z = [[model.add_var() for j in range(m)] for i in range(n)]
r = [[model.add_var(var_type="CONTINUOUS") for j in range(m)] for i in range(n)]

# Целевая функция
model.objective = maximize(rho)

# Ограничения


for i in range(n):
    for j in range(m):


        model += x[i][j] <= y[i]  # (4)
        model += x[i][j] <= 1  # (5)

        model += b[j]*x[i][j] - c[i][j]*x[i][j] - r[i][j] - z[i][j] >= 0  # (6)
        model += c[i][j] * x[i][j] <= c[i][j] + (1 - y[i]) * W  # (7)

        model += (1 - x[i][j]) * W + z[i][j] >= p  # (9)
        model += (1 - x[i][j]) * W + p >= z[i][j]  # (10)
        model += z[i][j] <= x[i][j] * W  # (11)
        model += z[i][j] >= 0  # (12)
        model += (1 - x[i][j]) * W + r[i][j] >= rho  # (13)
        model += (1 - x[i][j]) * W + rho >= r[i][j]  # (14)
        model += r[i][j] <= x[i][j] * W  # (15)
        model += r[i][j] >= 0  # (16)

# Решение задачи
model.optimize()
print(model.objective_value)
