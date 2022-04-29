import math
import numpy

from collections import Counter


def calcular_raios(train_x, train_y):
    e = 1e-20
    raios = []

    for i in range(len(train_x)):
        newData = train_x.copy()
        newData.pop(i)
        newData_y = train_y.copy()
        newData_y.pop(i)

        results = []

        for j in range(len(newData)):
            r = 0

            for k in range(len(train_x[i])):
                r += (train_x[i][k] - newData[j][k]) ** 2  # Distância Euclidiana

            results.append(math.sqrt(r))

        indexes = numpy.argsort(results)  # retorna os índices ordenados

        aux = 0
        while train_y[i] == newData_y[indexes[aux]]:
            aux += 1

        raios.append(results[indexes[aux]] - e)

    return raios


def _knn_improve(train_x, train_y, test, k, raios):
    results = []

    for i in range(len(train_x)):
        r = 0

        for j in range(len(test)):
            r += (test[j] - train_x[i][j]) ** 2  # Distância Euclidiana

        results.append(math.sqrt(r) / raios[i])  # Distância Euclidiana / Raio

    indexes = numpy.argsort(results)  # retorna os índices ordenados

    indexes = indexes[0:k]  # Pega os k índices mais próximos

    res = [train_y[i] for i in indexes]  # Retorna a classe de cada um dos vizinhos

    final = Counter(res)

    return final.most_common(1)[0][0]  # retorna a classe com maior frequência
