import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = fetch_california_housing()
x, t = data.data, data.target

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2, random_state=42)

def stand_data(X, method=1):
    if method == 1:
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        return (X - X_min) / (X_max - X_min)
    elif method == 2:
        a = -1
        b = 1
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        return a + (X - X_min) / (X_max - X_min) * (b - a)
    elif method == 3:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std
    else:
        print("Ошибка: Неправильно выбран метод стандартизации.")
        return None

x_train_stand = stand_data(x_train)
x_test_stand = stand_data(x_test)

basis_functions = [lambda x: 1]
basis_functions.extend([lambda x, i=i: x[i] for i in range(len(x_train_stand[0]))])

def getF(x, basis_functions):
    F = np.zeros((len(x), len(basis_functions)))
    for i in range(len(x)):
        for j in range(len(basis_functions)):
            F[i][j] = basis_functions[j](x[i])
    return F

def getW(F, t, alpha):
    w = np.linalg.inv((F.T @ F + alpha * np.identity(len(F[0])))) @ F.T @ t
    return w

def getY(F, w):
    return F @ w

def getE(Y, t):
    result = (1 / len(t)) * np.sum((t - Y) ** 2)
    return result

def gradientDescent(x_train, t_train, basis_functions, alpha, lr, max_iter, tol):
    np.random.seed(42)
    w = np.random.normal(0, 0.1, len(basis_functions))

    errors = []

    for iter in range(max_iter):
        F = getF(x_train, basis_functions)
        Y = getY(F, w)

        grad = -(t_train.T @ F).T + (w.T @ (F.T @ F)).T + alpha * w.T

        w -= lr * grad

        # Проверка условий остановки
        if np.linalg.norm(grad) < tol:
            print("Остановка по норме градиента")
            break

        if iter > 0 and abs(errors[-1] - getE(Y, t_train)) < tol:
            print("Остановка по изменению ошибки")
            break

        errors.append(getE(Y, t_train))

    return w, errors

alpha = 0.1
lern_rate = 1e-6
max_iter = 1000
tol = 1e-5

w, errors = gradientDescent(x_train_stand, t_train, basis_functions, alpha, lern_rate, max_iter, tol)

# Отображение графика зависимости ошибки от номера итерации
plt.plot(range(len(errors)), errors)
plt.xlabel('Номер итерации')
plt.ylabel('Ошибка')
plt.title('График зависимости ошибки от номера итерации')
plt.show()

# Получение матриц базисных функций для тестовой выборки
F_test = getF(x_test_stand, basis_functions)

# Получение предсказанных значений для тестовой выборки
Y_test = getY(F_test, w)

# Вывод ошибок на обучающей и тестовой выборках
print("Ошибка на обучающей выборке:", getE(getY(getF(x_train_stand, basis_functions), w), t_train))
print("Ошибка на тестовой выборке:", getE(Y_test, t_test))
