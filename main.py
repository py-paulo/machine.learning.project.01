from typing import Tuple

from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

from utils import calcular_raios, _knn_improve

import pandas as pd


class Classifier:

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.dataset, self.number_columns = self.__load_data()

    def __load_data(self) -> Tuple[TextFileReader or DataFrame, int]:
        dataset = pd.read_csv(self.data_path, header=None)
        dataset.sample(random_state=1)
        return dataset, len(dataset.columns)

    def tree(
            self,
            distance: int,
            index_class: int,
            test_size: int = 0.2,
            criterion: str = 'entropy'
    ) -> Tuple[float, list, list]:
        return self.__core('tree', distance=distance, index_class=index_class, test_size=test_size, criterion=criterion)

    def knn(
            self,
            distance: int,
            index_class: int,
            test_size: int = 0.2,
            metric: str = 'euclidean',
            algorithm: str = 'brute'
    ) -> Tuple[float, list, list]:
        return self.__core(
            'knn', distance=distance, index_class=index_class, test_size=test_size, metric=metric, algorithm=algorithm)

    def knn_improve(
            self,
            distance: int,
            index_class: int,
            test_size: int = 0.2,
            metric: str = 'euclidean',
            algorithm: str = 'brute'
    ) -> Tuple[float, list, list]:
        return self.__core(
            'knn-improve', distance=distance, index_class=index_class, test_size=test_size, metric=metric, algorithm=algorithm)

    def __core(
            self,
            classifier: str,
            distance: int,
            index_class: int,
            test_size: int = 0.2,
            metric: str = 'euclidean',
            algorithm: str = 'brute',
            criterion: str = 'entropy'
    ) -> Tuple[float, list, list]:
        y = self.dataset[index_class]
        if index_class == 0:
            X = self.dataset.loc[:, 1:self.number_columns - 1]
        else:
            X = self.dataset.loc[:, 0:self.number_columns - 1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=None, stratify=y)

        if classifier == 'knn':
            model = KNeighborsClassifier(
                n_neighbors=distance, metric=metric, algorithm=algorithm)
        elif classifier == 'tree':
            model = tree.DecisionTreeClassifier(criterion=criterion)
        elif classifier == 'knn-improve':
            train_x = X_train.values.tolist()
            train_y = y_train.values.tolist()

            test_x = X_test.values.tolist()
            test_y = y_test.values.tolist()
            result_knn_improve = []

            raios = calcular_raios(train_x, train_y)

            for i in range(len(test_x)):
                class_knn = _knn_improve(train_x, train_y, test_x[i], distance, raios)
                result_knn_improve.append(class_knn)

            acc = metrics.accuracy_score(result_knn_improve, test_y)
            show = round(acc * 100)

            return show, result_knn_improve, test_y

        else:
            raise Exception("Algoritmo escolhido não implementado")

        model = model.fit(X_train, y_train)
        result = model.predict(X_test)
        acc = metrics.accuracy_score(result, y_test)
        show = round(acc * 100)

        return show, result, y_test


if __name__ == '__main__':
    datasets = {
        'hill-valley': ('datasets/hill-valley/hill-valley.data', 100),
        'iris': ('datasets/iris/iris.data', 0),
        'wine': ('datasets/wine/wine.data', 0),
        'raisin': ('datasets/raisin/raisin.data', 0),
        'abalone': ('datasets/abalone/abalone.data', 0),
        'glass': ('datasets/glass/glass.data', 0),
        # 'accelerometer': ('datasets/accelerometer/accelerometer.data', 0)
    }

    print("--- kNN\n")
    for key, value in datasets.items():
        cf = Classifier(value[0])
        print(f"Base: {key}")
        for metric_name in ['euclidean', 'manhattan', 'chebyshev']:
            for index in range(1, 4):
                result, _, _ = cf.knn(index, value[1], metric=metric_name)
                print(
                    f"  - Distancia: {index}\n\tMetrica: {metric_name}\n\tResultado: {result}%")
            print()

    print("--- kNN Improve\n")
    for key, value in datasets.items():
        cf = Classifier(value[0])
        print(f"Base: {key}")

        for index in range(1, 4):
            result, _, _ = cf.knn(index, value[1])
            print(
                f"  - Distancia: {index}\n\tResultado: {result}%")
        print()

    print("--- Tree\n")
    for key, value in datasets.items():
        cf = Classifier(value[0])
        print(f"Base: {key}")
        for criterion in ['gini', 'entropy']:
            for index in range(1, 4):
                result, _, _ = cf.tree(index, value[1], criterion=criterion)
                print(
                    f"  - Distancia: {index}\n\tCritério: {criterion}\n\tResultado: {result}%")
            print()
