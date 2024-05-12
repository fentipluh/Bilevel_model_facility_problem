import numpy as np

# Здесь задаются параметры
n = 100  # Количество строк матрицы
m = 100  # Количество столбцов матрицы и размер массива b

# Используем np.random.randint для генерации целочисленных значений
# np.random.randint(low, high=None, size=None, dtype=int)
# low - наименьшее значение, high - наибольшее, size - размер вывода.
A = np.random.randint(0, 500, size=(n, m))  # Генерация матрицы размером n на m

b = np.random.randint(0, 500, size=m)  # Генерация массива b

# Путь к файлу на вывод
file_path = 'C:/Users/Fentipluh/PycharmProjects/diploma/dataset/gen_100_100_2.txt'

with open(file_path, 'w') as file:
    # Запись значений n и m
    file.write(f"{n} {m}\n")

    # Запись матрицы A
    for row in A:
        file.write(' '.join(map(str, row)) + '\n')

    # Запись массива b
    file.write(' '.join(map(str, b)) + '\n')

print(f"Файл {file_path} успешно записан.")
