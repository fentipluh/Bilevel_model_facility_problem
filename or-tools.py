from mip import Model, xsum, INTEGER

# Creating a model
model = Model()

# Defining variables x1, x2, x3
x1 = model.add_var(var_type=INTEGER)
x2 = model.add_var(var_type=INTEGER)
x3 = model.add_var(var_type=INTEGER)

# Defining auxiliary variable y for the product x1 * x2 * x3
y = model.add_var(var_type=INTEGER)

# Adding constraints to represent the product of x1, x2, and x3
model += y >= x1 + x2 + x3 - 2
model += y <= x1
model += y <= x2
model += y <= x3

# Solving the model
model.optimize()

# Retrieving and printing the optimal values
print("x1 =", x1.x)
print("x2 =", x2.x)
print("x3 =", x3.x)
print("y =", y.x)
