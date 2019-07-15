# Audio Models Stattus4
Audio Models and statistical tools used in Devs for Stattus4 Start-Up of Water Wasting Detection

## Statistical Analysis

- TODO: organizing and giving a coherent walkthrough

## Filtering

### Spectral

- Simple one based on mean mass of spectral frequencies.
- TODO: General Spectral Filtering function that allows to determines which statistics to base your filtering and to choose general Frequency-Time projection.

### Inverse Filtering

- Inverse Filtering with Tensorflow Optimization and Stochastic Gradient Descent

## Audio Deep Learning

- Deep Neural Networks for Tagging Audio based on binary tagged database using Audio Features Models and expert geophonist
- TODO: Road-Map Milestones:
0- implementar unidades, tanto para Time-series como para Spect FT
1- escolher um conjunto de parametros (tamanho de filtro da convolução, número de layers, tamanho do batch de dados por atualização da apredizagem). Implementar camada de sigmoids no final deste bulk de hidden layers.
2- Validação do funcionamento do modelo com poucos dados triviais (ruido de duas frequencias diferentes a principio).
3- Poe pra torar o cross-validation do conjunto escolhido no protocolo 1.
4- intervalo de tantos em tantos tempo pra verificar se não está havendo anormalidade na curva de acurácia. Adaptar learning rate se necessário.
5- escolher modelo com maior acurácia.

## Installation

```sh
pip install audiomodels
```
