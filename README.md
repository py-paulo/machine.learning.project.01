# Tópicos Especiais (Machine Learning)

Projeto 1

Objetivo: Comparar os resultados de algoritmos de Machine Learning com problemas de classificação. Escolher dois problemas de classificação (bases) do UCI: https://archive.ics.uci.edu/ml/index.php 
Algoritmos a serem testados: 
* Treinar 2 árvores de decisão com o criterion = “gini” e “entropy” - Treinar 6 kNN. Usar 2 medidas de distância diferentes e 3 tamanhos de vizinhança. 
* Treinar 3 kNN Improve. Usar 3 tamanhos de vizinhança. 
Protocolo Experimental: 
* Dividir o conjunto de dados em 80% para treinamento e 20% para testes. Treinar TODOS os algoritmos com os mesmos dados. 
Relatório: 
* Exibir os resultados (Taxas de Acerto) de todos os algoritmos. 

## Resultados

**Bases utilizadas:**
* [abalone](https://archive.ics.uci.edu/ml/datasets/abalone)
* [glass](https://archive.ics.uci.edu/ml/datasets/glass+identification)
* [hill-valley](http://archive.ics.uci.edu/ml/datasets/hill-valley)
* [iris](https://archive.ics.uci.edu/ml/datasets/iris)
* [raisin](https://archive.ics.uci.edu/ml/datasets/Raisin+Dataset)
* [wine](https://archive.ics.uci.edu/ml/datasets/wine)

```
--- kNN
Base: hill-valley
  - Distancia: 1
	Metrica: euclidean
	Resultado: 64%
  - Distancia: 2
	Metrica: euclidean
	Resultado: 54%
  - Distancia: 3
	Metrica: euclidean
	Resultado: 52%
  - Distancia: 1
	Metrica: manhattan
	Resultado: 56%
  - Distancia: 2
	Metrica: manhattan
	Resultado: 54%
  - Distancia: 3
	Metrica: manhattan
	Resultado: 56%
  - Distancia: 1
	Metrica: chebyshev
	Resultado: 37%
  - Distancia: 2
	Metrica: chebyshev
	Resultado: 40%
  - Distancia: 3
	Metrica: chebyshev
	Resultado: 37%
Base: iris
  - Distancia: 1
	Metrica: euclidean
	Resultado: 93%
  - Distancia: 2
	Metrica: euclidean
	Resultado: 93%
  - Distancia: 3
	Metrica: euclidean
	Resultado: 100%
  - Distancia: 1
	Metrica: manhattan
	Resultado: 100%
  - Distancia: 2
	Metrica: manhattan
	Resultado: 93%
  - Distancia: 3
	Metrica: manhattan
	Resultado: 93%
  - Distancia: 1
	Metrica: chebyshev
	Resultado: 97%
  - Distancia: 2
	Metrica: chebyshev
	Resultado: 93%
  - Distancia: 3
	Metrica: chebyshev
	Resultado: 93%
Base: wine
  - Distancia: 1
	Metrica: euclidean
	Resultado: 75%
  - Distancia: 2
	Metrica: euclidean
	Resultado: 64%
  - Distancia: 3
	Metrica: euclidean
	Resultado: 72%
  - Distancia: 1
	Metrica: manhattan
	Resultado: 92%
  - Distancia: 2
	Metrica: manhattan
	Resultado: 81%
  - Distancia: 3
	Metrica: manhattan
	Resultado: 81%
  - Distancia: 1
	Metrica: chebyshev
	Resultado: 72%
  - Distancia: 2
	Metrica: chebyshev
	Resultado: 72%
  - Distancia: 3
	Metrica: chebyshev
	Resultado: 78%
Base: raisin
  - Distancia: 1
	Metrica: euclidean
	Resultado: 77%
  - Distancia: 2
	Metrica: euclidean
	Resultado: 83%
  - Distancia: 3
	Metrica: euclidean
	Resultado: 78%
  - Distancia: 1
	Metrica: manhattan
	Resultado: 73%
  - Distancia: 2
	Metrica: manhattan
	Resultado: 79%
  - Distancia: 3
	Metrica: manhattan
	Resultado: 84%
  - Distancia: 1
	Metrica: chebyshev
	Resultado: 73%
  - Distancia: 2
	Metrica: chebyshev
	Resultado: 82%
  - Distancia: 3
	Metrica: chebyshev
	Resultado: 82%
Base: abalone
  - Distancia: 1
	Metrica: euclidean
	Resultado: 49%
  - Distancia: 2
	Metrica: euclidean
	Resultado: 50%
  - Distancia: 3
	Metrica: euclidean
	Resultado: 56%
  - Distancia: 1
	Metrica: manhattan
	Resultado: 49%
  - Distancia: 2
	Metrica: manhattan
	Resultado: 47%
  - Distancia: 3
	Metrica: manhattan
	Resultado: 52%
  - Distancia: 1
	Metrica: chebyshev
	Resultado: 47%
  - Distancia: 2
	Metrica: chebyshev
	Resultado: 49%
  - Distancia: 3
	Metrica: chebyshev
	Resultado: 53%
Base: glass
  - Distancia: 1
	Metrica: euclidean
	Resultado: 100%
  - Distancia: 2
	Metrica: euclidean
	Resultado: 100%
  - Distancia: 3
	Metrica: euclidean
	Resultado: 98%
  - Distancia: 1
	Metrica: manhattan
	Resultado: 100%
  - Distancia: 2
	Metrica: manhattan
	Resultado: 95%
  - Distancia: 3
	Metrica: manhattan
	Resultado: 100%
  - Distancia: 1
	Metrica: chebyshev
	Resultado: 100%
  - Distancia: 2
	Metrica: chebyshev
	Resultado: 98%
  - Distancia: 3
	Metrica: chebyshev
	Resultado: 100%
--- kNN Improve
Base: hill-valley
  - Distancia: 1
	Resultado: 62%
  - Distancia: 2
	Resultado: 51%
```