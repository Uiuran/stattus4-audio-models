# Audio Models Stattus4
Audio Models and statistical tools used in Devs for Stattus4 Start-Up of Water Wasting Detection

Note: this is a Alpha Version.

## Statistical Analysis

## Filtering

### Spectral

### Inverse Filtering

- Inverse Filtering with Tensorflow Optimization and Stochastic Gradient Descent

## Audio Deep Learning

Milestones
02/09/2018 -
- Classificador pronto para estrutura de dados(imagem) de qualquer tamanho. Done
- Modelo pronto para Framing Diverso. Done
- Modelo Pronto para Hyperparameter tuning. Done
- Builder de modelo final. Fazendo
  -- Uso de looping já feito **na mão**, apenas colocar verificador de tamanho do canal para adicionar automaticamente filterbanks maiores ou mais camadas, de modo que a arquitetura se adapte ao tamanho do frame.

- Deep Neural Networks for Tagging Audio based on binary tagged database using Audio Features Models.
- TODO: Road-Map Milestones:
 -- 0. Implementar unidades. Done
 -- 1. escolher um conjunto de parametros (tamanho de filtro da convolução, número de layers, tamanho do batch de dados por atualização da apredizagem). Done
 -- 2. Validação do funcionamento do modelo com poucos dados triviais (ruido de duas frequencias diferentes a principio). Overfit Done.
 -- 3. Cross-validation.
 -- 5. Neural Network Selection Criteria.

## Installation
With conda installed:
```python
git clone https://github.com/Uiuran/stattus4-audio-models.git
cd ~/path/to/dir/stattus4-audio-models/
conda activate env
pip install .
``` 
