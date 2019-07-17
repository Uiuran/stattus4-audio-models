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
 - 0. Implementar unidades.
 - 1. escolher um conjunto de parametros (tamanho de filtro da convolução, número de layers, tamanho do batch de dados por atualização da apredizagem).
 - 2. Validação do funcionamento do modelo com poucos dados triviais (ruido de duas frequencias diferentes a principio).
 - 3. Cross-validation.
 - 4. Check-pointing.
 - 5. Accuracy | Confusion Matrix | Business criteria choosing.

## Installation

```sh
pip install audiomodels
```
