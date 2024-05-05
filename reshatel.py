from itertools import product
from sys import stdout as out
from mip import *
import numpy as np


with open('input.txt', 'r') as file:
    data = file.readline().strip().split()
    n = int(data[0])
    m = int(data[1])

f= [15] * n
V = 30

c = np.loadtxt('input_c.txt')
b = np.loadtxt('input_b.txt')

difference = b - c

# Находим максимальное значение по всем элементам полученной матрицы
W = np.max(difference)

print("Максимальное значение (W) =", W)

model = Model(sense=MAXIMIZE, solver_name=CBC)
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

#model += xsum(z[i][j] - (f[i]*y[i]) for i in range (n) for j in range(m)) >= V #(2)
#model += xsum((z[i][j] for i in range (n) for j in range (m)) - ((f[i]*y[i]) for i in range (n) for j in range (m)))
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

model.optimize()
print(model.objective_value)
print("Значения вектора y:")
for var in model.vars:
    if var.name[0] == "y":
        print(f"{var.name} = {var.x}")

